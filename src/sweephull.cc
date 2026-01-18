// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cmath>
#include <limits>

#include <ATen/TensorAccessor.h>

#include <torch_delaunay/predicate.h>
#include <torch_delaunay/sweephull.h>
#include <torch_delaunay/triangle.h>


namespace torch_delaunay {


inline torch::Tensor
euclidean_distance2d(const torch::Tensor& p, const torch::Tensor& q)
{
    return torch::sqrt((p - q).square().sum(/*dim=*/1));
}

template <typename scalar_t>
double
euclidean_distance2d_kernel(
    const at::TensorAccessor<scalar_t, 1>& p, const at::TensorAccessor<scalar_t, 1>& q
)
{
    const auto d0 = double(q[0]) - double(p[0]);
    const auto d1 = double(q[1]) - double(p[1]);
    return std::sqrt(d0 * d0 + d1 * d1);
}


/// An operation to compute Delaunay-triangulation using sweep-hull algorithm.
template <typename scalar_t>
struct shull {
    using triangle_type = std::tuple<int64_t, int64_t, int64_t>;

    using point_type = at::TensorAccessor<scalar_t, 1>;
    using center_type = at::TensorAccessor<double, 1>;
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

    const center_type m_center_ptr;
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
      m_center_ptr(center.accessor<double, 1>()),
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
        const auto dx = static_cast<double>(m_points_ptr[i][0]) - m_center_ptr[0];
        const auto dy = static_cast<double>(m_points_ptr[i][1]) - m_center_ptr[1];

        const auto rad = dx / (std::abs(dx) + std::abs(dy));
        const auto angle = (dy > 0.0 ? 3.0 - rad : 1.0 + rad) / 4.0;

        const auto k = std::llround(std::floor(angle * static_cast<double>(hash.size())));
        return static_cast<std::size_t>(k) % hash.size();
    }

    void
    insert_visible_edge(int64_t i0, int64_t i1)
    {
        TORCH_CHECK(i0 != i1, "shull2d: encountered a hull loop");
        prev[i1] = i0;
        next[i0] = i1;

        hash[hash_key(i0)] = i0;
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

    /// Returns a triangle orientation.
    ///
    /// A positive number is returned, when specified points are ordered counterclockwise,
    /// a negative is returned, when points are ordered clockwise. Zero is returned in case
    /// if points are collinear.
    ///
    /// In other words, method returns true when p2 lies on to the left of (p0, p1) vector.
    inline static scalar_t
    orient(const point_type& p0, const point_type& p1, const point_type& p2)
    {
        return orient2d_kernel<scalar_t>(p0, p1, p2);
    }

    /// See \ref orient.
    inline static scalar_t
    orient(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
    {
        return orient(
            p0.accessor<scalar_t, 1>(), p1.accessor<scalar_t, 1>(), p2.accessor<scalar_t, 1>()
        );
    }

    /// See \ref orient.
    inline scalar_t
    orient(int64_t i0, int64_t i1, int64_t i2) const
    {
        const auto [p0, p1, p2] = get_tri(i0, i1, i2);
        return orient(p0, p1, p2);
    }

    /// See \ref orient.
    inline scalar_t
    orient(const triangle_type& tri) const
    {
        return orient(std::get<0>(tri), std::get<1>(tri), std::get<2>(tri));
    }

    /// Returns true, when the specified triangle (simplex) is collinear.
    ///
    /// Here we use an extended interpretation of collinear points. Apart from considering points
    /// that reside on the same line we also validate that the circumscribed radius is larger than
    /// a specified epsilon.
    static bool
    iscollinear(
        const point_type& p0,
        const point_type& p1,
        const point_type& p2,
        std::optional<scalar_t> eps
    )
    {
        auto radius = circumradius2d_kernel<scalar_t>(p0, p1, p2);
        return std::isnan(radius) || std::isinf(radius)
               || (eps.has_value() && radius < eps.value());
    }

    /// Returns true, when the specified triangle is collinear.
    static bool
    iscollinear(
        const torch::Tensor& p0,
        const torch::Tensor& p1,
        const torch::Tensor& p2,
        std::optional<scalar_t> eps
    )
    {
        return iscollinear(
            p0.accessor<scalar_t, 1>(), p1.accessor<scalar_t, 1>(), p2.accessor<scalar_t, 1>(), eps
        );
    }

    /// Returns true, when the specified triangle is collinear.
    bool
    iscollinear(int64_t i0, int64_t i1, int64_t i2, std::optional<scalar_t> eps) const
    {
        const auto [p0, p1, p2] = get_tri(i0, i1, i2);
        return iscollinear(p0, p1, p2, /*eps=*/eps);
    }

    static bool
    isduplicate(const point_type& p0, const point_type& p1, std::optional<scalar_t> eps)
    {
        return (
            eps.has_value() && std::abs(p0[0] - p1[0]) <= eps.value()
            && std::abs(p0[1] - p1[1]) <= eps.value()
        );
    }

    inline bool
    isduplicate(int64_t i0, int64_t i1, std::optional<scalar_t> eps) const
    {
        return isduplicate(m_points_ptr[i0], m_points_ptr[i1], eps);
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
        return std::make_tuple(m_points_ptr[i0], m_points_ptr[i1], m_points_ptr[i2]);
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
        int64_t halfedges_size = static_cast<int64_t>(halfedges.size());
        TORCH_CHECK(a <= halfedges_size, "shull2d: encountered wrong half-edge: ", a, " -> ", b);

        if (a < halfedges_size) {
            halfedges[a] = b;
        }
        if (a == halfedges_size) {
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

    void
    forward(const triangle_type& triangle, int64_t edge)
    {
        auto [j, i, k] = triangle;
        tri[i] = edge;
        next[j] = j;
    }

    void
    backward(const triangle_type& triangle, int64_t edge)
    {
        auto [j, i, k] = triangle;
        tri[j] = edge;
        next[k] = k;
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
    using hull_type = shull<scalar_t>;

    // The seed of the triangulat might be empty in case when all points are
    // comprising only collinear triangles.
    auto points_out = torch::empty({0, 2}, points.options());

    auto indices_options = torch::TensorOptions().dtype(torch::kInt64);
    auto indices_out = torch::empty({0, 3}, indices_options);

    // Find the first seed point with the minimum distance to a centroid of the point cloud.
    auto [min, max] = points.aminmax(0);
    auto dists = euclidean_distance2d(points, (max + min) / 2);

    auto [_, indices] = at::topk(dists, 1, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    auto index0 = indices[0].item<int64_t>();
    auto point0 = points[index0];
    const auto p0 = point0.accessor<scalar_t, 1>();

    // Find the second seed point with the minimum distance (larger than 0)
    // to the first point of the seed triangle.
    int64_t index1 = -1;
    auto dist_min = std::numeric_limits<double>::infinity();

    for (int64_t i = 0; i < dists.size(0); i++) {
        if (i == index0) {
            continue;
        }

        auto pointi = points[i];
        auto pi = pointi.accessor<scalar_t, 1>();
        auto dist = euclidean_distance2d_kernel(p0, pi);

        if (dist > 0 && dist < dist_min) {
            index1 = i;
            dist_min = dist;
        }
    }
    // When all points collapse to a single point, return an empty seed triangle.
    if (index1 == -1) {
        return std::make_tuple(points_out, indices_out);
    }

    auto point1 = points[index1];
    const auto p1 = point1.accessor<scalar_t, 1>();

    // Find the third point such that it forms the smallest circumcircle with p0 and p1.
    int64_t index2 = -1;
    auto radius_min = std::numeric_limits<double>::infinity();

    for (int64_t i = 0; i < points.size(0); i++) {
        if (i == index0 || i == index1) {
            continue;
        }

        auto pointi = points[i];
        auto pi = pointi.accessor<scalar_t, 1>();
        auto radius = circumradius2d_kernel(p0, p1, pi);

        if (radius > 0 && radius < radius_min) {
            if (!hull_type::iscollinear(p0, p1, pi, /*eps=*/eps)) {
                index2 = i;
                radius_min = radius;
            }
        }
    }
    // There are no non-collinear simplices, therefore return empty tensors in the result.
    if (index2 == -1) {
        return std::make_tuple(points_out, indices_out);
    }

    auto point2 = points[index2];
    const auto p2 = point2.accessor<scalar_t, 1>();

    if (hull_type::orient(p0, p1, p2) < 0) {
        std::swap(index1, index2);
        std::swap(point1, point2);
    }

    points_out = at::vstack({point0.view({1, 2}), point1.view({1, 2}), point2.view({1, 2})});
    indices_out = torch::tensor({index0, index1, index2}, indices_options);

    return std::make_tuple(points_out, indices_out);
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
    const auto p0 = triangle[0], p1 = triangle[1], p2 = triangle[2];
    const auto center = circumcenter2d_kernel<scalar_t>(
        p0.template accessor<scalar_t, 1>(), p1.template accessor<scalar_t, 1>(),
        p2.template accessor<scalar_t, 1>()
    );

    const auto ordering = at::argsort(euclidean_distance2d(points, center));
    const auto ordering_ptr = ordering.template accessor<int64_t, 1>();

    const auto n = points.size(0);
    shull<scalar_t> hull(n, center, points);

    auto indices_ptr = indices.template accessor<int64_t, 1>();
    auto i0 = indices_ptr[0];
    auto i1 = indices_ptr[1];
    auto i2 = indices_ptr[2];

    hull.insert_visible_edge(i0, i1);
    hull.insert_visible_edge(i1, i2);
    hull.insert_visible_edge(i2, i0);
    hull.start = i0;

    hull.tri[i0] = 0;
    hull.tri[i1] = 1;
    hull.tri[i2] = 2;

    hull.push_tri(i0, i1, i2);
    hull.push_edges(-1, -1, -1);

    for (const auto k : c10::irange(n)) {
        auto i = ordering_ptr[k];
        // Skip points comprising a seed triangle.
        if (i == i0 || i == i1 || i == i2) {
            continue;
        }
        if (k > 0 && hull.isduplicate(ordering_ptr[k - 1], i, eps)) {
            continue;
        }

        auto is = hull.find_visible_edge(i);
        // TODO: Make sure what we found is on the hull?
        auto ie = hull.prev[is];

        while (ie != -1 && hull.orient(i, ie, hull.next[ie]) > 0) {
            ie = hull.next[ie] == hull.prev[is] ? -1 : hull.next[ie];
        }
        // After iterating over the hull, there are no triangles found that are
        // oriented clounterclockwise, skipping this points from the triangulation.
        if (ie == -1) {
            continue;
        }

        // Ensure that each triangle's circumradius is at least the size of the specified
        // epsilon value, skip the point from triangulation otherwise.
        if (hull.iscollinear(ie, i, hull.next[ie], /*eps=*/eps)) {
            continue;
        }

        auto edge0 = hull.push_tri(ie, i, hull.next[ie]);
        auto edge1 = hull.push_edges(-1, -1, hull.tri[ie]);
        hull.tri[ie] = edge0;
        hull.tri[i] = edge1;

        // Traverse forward through the hull, adding more triangles and flipping
        // them recursively.
        auto in = hull.next[ie];

        while (hull.orient(i, in, hull.next[in]) < 0) {
            const auto tri = std::make_tuple(in, i, hull.next[in]);
            in = hull.next[in];

            edge0 = hull.push_tri(tri);
            edge1 = hull.push_forward_edges(tri);
            hull.forward(tri, edge1);
        }

        // Traverse backward through the hull, adding more triangles and flipping
        // them recursively.
        if (hull.prev[is] == ie) {
            while (hull.orient(i, hull.prev[ie], ie) < 0) {
                const auto tri = std::make_tuple(hull.prev[ie], i, ie);
                ie = hull.prev[ie];

                edge0 = hull.push_tri(tri);
                edge1 = hull.push_backward_edges(tri);
                hull.backward(tri, edge0);
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
