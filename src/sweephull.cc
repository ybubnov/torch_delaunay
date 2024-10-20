/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <cmath>

#include <ATen/TensorAccessor.h>

#include <torch_delaunay/predicate.h>
#include <torch_delaunay/sweephull.h>
#include <torch_delaunay/triangle.h>


namespace torch_delaunay {


inline torch::Tensor
euclidean_distance2d(const torch::Tensor& p, const torch::Tensor& q)
{
    return torch::linalg::norm(p - q, /*ord=*/2, /*dim=*/1, false, c10::nullopt);
}


/// An operation to compute Delaunay-triangulation using sweep-hull algorithm.
template <typename scalar_t>
struct shull {
    using triangle_type = std::tuple<int64_t, int64_t, int64_t>;

    using point_type = at::TensorAccessor<scalar_t, 1>;
    using point_array_type = at::TensorAccessor<scalar_t, 2>;

    /// Hash stores radially-sorted edges, and used to query a visible vertex.
    std::vector<int64_t> hash;

    /// Triangles are used to store the final triangulation simplices.
    std::vector<int64_t> triangles;
    std::vector<int64_t> halfedges;

    /// A common stack for `flip` operation, keep it here to reduce a number of
    /// de-allocations, also use vector as an underlying storage.
    std::stack<int64_t, std::vector<int64_t>> unvisited_edges;

    std::vector<int64_t> tri;

    /// These vectors provide an access to the convex hull of the triangulation,
    /// by implementing a double-linked list of vertices.
    std::vector<int64_t> next;
    std::vector<int64_t> prev;

    const point_type m_center_ptr;
    const point_array_type m_points_ptr;

    int64_t start;

    shull(int64_t n, const torch::Tensor& center, const torch::Tensor& points)
    : hash(),
      triangles(),
      halfedges(),
      unvisited_edges(),
      tri(n, -1),
      next(n, -1),
      prev(n, -1),
      m_center_ptr(center.accessor<scalar_t, 1>()),
      m_points_ptr(points.accessor<scalar_t, 2>()),
      start(0)
    {
        auto hash_size = std::llround(std::ceil(std::sqrt(n)));
        hash.resize(static_cast<std::size_t>(hash_size));
        std::fill(hash.begin(), hash.end(), -1);

        std::size_t max_triangles = n < 3 ? 1 : 2 * n - 5;
        triangles.reserve(max_triangles);
        halfedges.reserve(max_triangles);
    }

    /// Computes the hash key for the given 2-dimensional point.
    ///
    /// The key represents pseudo-angle from the center of the triangulation.
    std::size_t
    hash_key(int64_t i) const
    {
        const auto dx = m_points_ptr[i][0] - m_center_ptr[0];
        const auto dy = m_points_ptr[i][1] - m_center_ptr[1];

        const auto rad = dx / (std::abs(dx) + std::abs(dy));
        const auto angle = (dy > 0.0 ? 3.0 - rad : 1.0 + rad) / 4.0;

        const auto k = std::llround(std::floor(angle * static_cast<scalar_t>(hash.size())));
        return static_cast<std::size_t>(k) % hash.size();
    }

    void
    insert_visible_edge(int64_t i, int64_t j)
    {
        next[i] = j;
        prev[j] = i;

        hash[hash_key(i)] = i;
    }

    int64_t
    find_visible_edge(int64_t i) const
    {
        const auto key = hash_key(i);
        int64_t edge_index = 0;

        for (std::size_t j = 0; j < hash.size(); j++) {
            edge_index = hash[(key + j) % hash.size()];

            if (edge_index != -1 && edge_index != next[edge_index]) {
                break;
            }
        }

        return edge_index;
    }

    int64_t
    push_tri(int64_t i0, int64_t i1, int64_t i2)
    {
        int64_t edge = triangles.size();
        triangles.push_back(i0);
        triangles.push_back(i1);
        triangles.push_back(i2);
        return edge;
    }

    int64_t
    push_tri(const triangle_type& triangle)
    {
        return push_tri(std::get<0>(triangle), std::get<1>(triangle), std::get<2>(triangle));
    }

    /// Returns true when specified points are ordered counterclockwise.
    ///
    /// In other words, method returns true when i2 lies on to the left of (i0, i1) vector.
    inline static bool
    ccw(point_type p0, point_type p1, point_type p2)
    {
        return orient2d_kernel<scalar_t>(p0, p1, p2) > scalar_t(0);
    }

    /// Returns true when specified points are ordered counterclockwise.
    inline static bool
    ccw(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
    {
        return ccw(
            p0.accessor<scalar_t, 1>(), p1.accessor<scalar_t, 1>(), p2.accessor<scalar_t, 1>()
        );
    }

    /// Returns true when specified points are ordered counterclockwise.
    inline bool
    ccw(int64_t i0, int64_t i1, int64_t i2) const
    {
        const auto [p0, p1, p2] = get_tri(i0, i1, i2);
        return ccw(p0, p1, p2);
    }

    /// Returns true when specified points are ordered counterclockwise.
    inline bool
    ccw(const triangle_type& tri) const
    {
        return ccw(std::get<0>(tri), std::get<1>(tri), std::get<2>(tri));
    }

    /// Returns true, when the specified triangle (simplex) is coplanar.
    ///
    /// Here we use an extended interpretation of coplanar points. Apart from considering points
    /// that reside on the same line we also validate that the circumscribed radius is larger than
    /// a specified epsilon.
    static bool
    iscoplanar(point_type p0, point_type p1, point_type p2, std::optional<scalar_t> eps)
    {
        auto radius = circumradius2d_kernel<scalar_t>(p0, p1, p2);
        return std::isnan(radius) || std::isinf(radius)
               || (eps.has_value() && radius < eps.value());
    }

    /// Returns true, when the specified triangle is coplanar.
    static bool
    iscoplanar(
        const torch::Tensor& p0,
        const torch::Tensor& p1,
        const torch::Tensor& p2,
        std::optional<scalar_t> eps
    )
    {
        return iscoplanar(
            p0.accessor<scalar_t, 1>(), p1.accessor<scalar_t, 1>(), p2.accessor<scalar_t, 1>(), eps
        );
    }

    /// Returns true, when the specified triangle is coplanar.
    bool
    iscoplanar(int64_t i0, int64_t i1, int64_t i2, std::optional<scalar_t> eps) const
    {
        const auto [p0, p1, p2] = get_tri(i0, i1, i2);
        return iscoplanar(p0, p1, p2, /*eps=*/eps);
    }

    inline bool
    incircle(int64_t edge0, int64_t edge1, int64_t edge2, int64_t edge3) const
    {
        auto sign = incircle2d_kernel<scalar_t>(
            m_points_ptr[triangles[edge0]], m_points_ptr[triangles[edge1]],
            m_points_ptr[triangles[edge2]], m_points_ptr[triangles[edge3]]
        );
        return sign < scalar_t(0);
    }

    /// Returns coordinates of the specified triangle.
    ///
    /// Method does not checks the boundaries of the specified indices, therefore, when
    /// indices are outside of the points bounds, the behaviour is undefined.
    const std::tuple<point_type, point_type, point_type>
    get_tri(int64_t i0, int64_t i1, int64_t i2) const
    {
        return std::forward_as_tuple(m_points_ptr[i0], m_points_ptr[i1], m_points_ptr[i2]);
    }

    int64_t
    push_edges(int64_t a, int64_t b, int64_t c)
    {
        int64_t edge = triangles.size() - 3;
        push_edge(edge, a);
        push_edge(edge + 1, b);
        push_edge(edge + 2, c);

        return flip(edge + 2);
    }

    int64_t
    push_forward_edges(const triangle_type& triangle)
    {
        const auto [j, i, k] = triangle;
        return push_edges(tri[i], -1, tri[j]);
    }

    int64_t
    push_backward_edges(const triangle_type& triangle)
    {
        const auto [j, i, k] = triangle;
        return push_edges(-1, tri[k], tri[j]);
    }

    inline void
    push_halfedge(int64_t a, int64_t b)
    {
        TORCH_CHECK(a <= halfedges.size(), "shull2d: encountered wrong half-edge: ", a, " -> ", b);

        if (a < halfedges.size()) {
            halfedges[a] = b;
        }
        if (a == halfedges.size()) {
            halfedges.push_back(b);
        }
    }

    void
    push_edge(int64_t a, int64_t b)
    {
        if (a != -1) {
            push_halfedge(a, b);
        }
        if (b != -1) {
            push_halfedge(b, a);
        }
    }

    triangle_type
    forward(const triangle_type& triangle, int64_t edge)
    {
        auto [j, i, k] = triangle;
        tri[i] = edge;

        std::swap(j, next[j]);
        TORCH_CHECK(j != next[j], "shull2d: encountered coplanar simplex");

        return std::make_tuple(j, i, next[j]);
    }

    triangle_type
    backward(const triangle_type& triangle, int64_t edge)
    {
        auto [j, i, k] = triangle;
        tri[j] = edge;
        next[k] = k;

        std::swap(k, prev[k]);
        TORCH_CHECK(k != prev[k], "shull2d: encountered coplanar simplex");

        return std::make_tuple(prev[k], i, k);
    }

    inline triangle_type
    tri_edges(int64_t edge) const
    {
        int64_t edge0 = 3 * (edge / 3);
        int64_t edge1 = edge0 + (edge + 1) % 3;
        int64_t edge2 = edge0 + (edge + 2) % 3;
        return std::make_tuple(edge, edge1, edge2);
    }

    void
    flip_tri(int64_t edge0, int64_t edge2)
    {
        auto end = start;
        do {
            if (tri[end] == edge2) {
                tri[end] = edge0;
                break;
            }

            end = prev[end];
        } while (end != start);
    }

    /// Flip triangle edges recursively until they all comply to Delaunay condition.
    ///
    /// \return The next edge of the triangle (in ccw direction).
    std::size_t
    flip(int64_t edge)
    {
        int64_t ar = -1, al = -1;
        int64_t br = -1, bl = -1;

        unvisited_edges.push(edge);

        while (unvisited_edges.size() > 0) {
            auto a = unvisited_edges.top();
            auto b = halfedges[a];

            std::tie(a, al, ar) = tri_edges(a);
            std::tie(b, br, bl) = tri_edges(b);

            unvisited_edges.pop();
            if (b == -1) {
                continue;
            }

            if (incircle(ar, a, al, bl)) {
                triangles[a] = triangles[bl];
                triangles[b] = triangles[ar];

                if (halfedges[bl] == -1) {
                    flip_tri(a, bl);
                }

                push_edge(a, halfedges[bl]);
                push_edge(b, halfedges[ar]);
                push_edge(ar, bl);

                unvisited_edges.push(br);
                unvisited_edges.push(a);
            }
        }

        return ar;
    }
};


template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor>
shull2d_seed_kernel(const torch::Tensor& points, std::optional<scalar_t> eps)
{
    // Indices of the seed triangle.
    torch::Tensor index0, index1, index2;
    torch::Tensor values, indices;
    torch::Tensor p0, p1, p2;

    // Choose seed points close to a centroid of the point cloud.
    auto [min, max] = points.aminmax(0);
    const auto centroid = (max + min) / 2;
    const auto dists = euclidean_distance2d(points, centroid);

    std::tie(values, indices) = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    index0 = indices[0];
    index1 = indices[1];

    p0 = points.index({index0}).view({-1, 2});
    p1 = points.index({index1}).view({-1, 2});

    // Find the third point such that forms the smallest circumcircle with i0 and i1.
    const auto radii = circumradius2d(p0, p1, points);
    std::tie(values, indices) = at::topk(radii, 3, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);

    // For points p0 and p1, radii of circumscribed circle will be set to `nan`, therefore
    // at 0 index will be a point with the minimum radius.
    std::size_t i = 0;
    for (; i < values.size(0); i++) {
        index2 = indices[i];
        p2 = points.index({index2}).view({-1, 2});

        if (!shull<scalar_t>::iscoplanar(p0[0], p1[0], p2[0], /*eps=*/eps)) {
            break;
        }
    }

    // There are no non-coplanar simplices, therefore return empty tensors in the result.
    if (i == values.size(0)) {
        auto points_out = torch::empty({0, 2}, points.options());
        auto indices_out = torch::empty({0, 3}, indices.options());

        return std::forward_as_tuple(points_out, indices_out);
    }

    if (shull<scalar_t>::ccw(p0[0], p1[0], p2[0])) {
        std::swap(index1, index2);
        std::swap(p1, p2);
    }

    auto points_out = at::vstack({p0, p1, p2});
    auto indices_out = at::vstack({index0, index1, index2});

    return std::forward_as_tuple(points_out, indices_out);
}


template <typename scalar_t>
torch::Tensor
shull2d_kernel(const torch::Tensor& points, std::optional<scalar_t> eps)
{
    TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got ", points.dim(), "D");
    TORCH_CHECK(points.size(0) >= 3, "shull2d expects at least 3 points, got ", points.size(0));
    TORCH_CHECK(points.size(1) == 2, "shull2d expects 2-dim points, got ", points.size(1), "-dim");

    // Find the seed triangle (comprised of vertices located approximately
    // at the center of the point cloud).
    const auto [triangle, indices] = shull2d_seed_kernel<scalar_t>(points, /*eps=*/eps);
    const auto triangle_options = points.options().dtype(torch::kInt64);

    if (triangle.size(0) == 0) {
        return torch::empty({0, 3}, triangle_options);
    }

    // Radially resort the points.
    const auto center = circumcenter2d(triangle.unsqueeze(0)).squeeze(0);
    const auto ordering = at::argsort(euclidean_distance2d(points, center));
    const auto ordering_ptr = ordering.template accessor<int64_t, 1>();

    const auto n = points.size(0);
    shull<scalar_t> hull(n, center, points);

    auto indices_ptr = indices.template accessor<int64_t, 2>();
    auto i0 = indices_ptr[0][0];
    auto i1 = indices_ptr[1][0];
    auto i2 = indices_ptr[2][0];

    hull.insert_visible_edge(i0, i1);
    hull.insert_visible_edge(i1, i2);
    hull.insert_visible_edge(i2, i0);
    hull.start = i0;

    hull.tri[i0] = 0;
    hull.tri[i1] = 1;
    hull.tri[i2] = 2;

    hull.push_tri(i0, i1, i2);
    hull.push_edges(-1, -1, -1);

    // 3 first points are already part of the triangulation, therefore they can be skipped.
    for (const auto k : c10::irange(3, n)) {
        auto i = ordering_ptr[k];
        auto is = hull.find_visible_edge(i);
        // TODO: Make sure what we found is on the hull?
        auto ie = hull.prev[is];

        while (hull.ccw(ie, i, hull.next[ie]) && ie != -1) {
            ie = hull.next[ie];
        }
        // TODO: what to do with this vertex: maybe raise an exception?
        if (ie == -1) {
            continue;
        }

        // Ensure that each triangle's circumradius is at least the size of the specified
        // epsilon value, skip the point from triangulation otherwise.
        if (hull.iscoplanar(ie, i, hull.next[ie], /*eps=*/eps)) {
            continue;
        }

        auto edge0 = hull.push_tri(ie, i, hull.next[ie]);
        auto edge1 = hull.push_edges(-1, -1, hull.tri[ie]);
        hull.tri[ie] = edge0;
        hull.tri[i] = edge1;

        // Traverse forward through the hull, adding more triangles and flipping
        // them recursively.
        auto in = hull.next[ie];
        auto tri = std::make_tuple(in, i, hull.next[in]);

        while (!hull.ccw(tri)) {
            edge0 = hull.push_tri(tri);
            edge1 = hull.push_forward_edges(tri);

            tri = hull.forward(tri, edge1);
            in = std::get<0>(tri);
        }

        // Traverse backward through the hull, adding more triangles and flipping
        // them recursively.
        if (hull.prev[is] == ie) {
            auto tri = std::make_tuple(hull.prev[ie], i, ie);
            while (!hull.ccw(tri)) {
                edge0 = hull.push_tri(tri);
                edge1 = hull.push_backward_edges(tri);

                tri = hull.backward(tri, edge0);
                ie = std::get<2>(tri);
            }
        }

        hull.start = ie;
        hull.insert_visible_edge(i, in);
        hull.insert_visible_edge(ie, i);
    }

    int64_t tn = hull.triangles.size() / 3;
    auto triangles = torch::tensor(std::move(hull.triangles), triangle_options);

    return triangles.view({tn, 3});
}


/// Compute seed triangle for the sweep-hull triangulation algorithm.
///
/// The method computes the center of all specified points and returns coordinates and indices
/// of the triangle comprised of points with the lowest distance from the center of point cloud.
torch::Tensor
shull2d(const torch::Tensor& points, std::optional<const at::Scalar> eps)
{
    torch::Tensor triangles;

    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "shull2d", [&] {
        std::optional<scalar_t> epsilon;
        if (eps.has_value()) {
            epsilon = eps.value().to<scalar_t>();
        }
        triangles = shull2d_kernel<scalar_t>(points, epsilon);
    });

    return triangles;
}


} // namespace torch_delaunay
