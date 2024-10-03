#include <cmath>

#include <torch_delaunay/predicate.h>
#include <torch_delaunay/sweephull.h>
#include <torch_delaunay/triangle.h>


namespace torch_delaunay {


inline torch::Tensor
euclidean_distance2d(const torch::Tensor& p, const torch::Tensor& q)
{
    return torch::linalg::norm(p - q, /*ord=*/2, /*dim=*/1, false, c10::nullopt);
}


inline bool
ccw(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    return orient2d(p0, p1, p2).gt(0).all().item<bool>();
}


torch::Tensor
circumcenter2d(const torch::Tensor& points)
{
    auto center
        = circumcenter2d(points[0].unsqueeze(0), points[1].unsqueeze(0), points[2].unsqueeze(0));
    return center.squeeze(0);
}


/// An operation to compute Delaunay-triangulation using sweep-hull algorithm.
struct shull {
    using triangle_type = std::tuple<int64_t, int64_t, int64_t>;

    /// Hash stores radially-sorted edges, and used to query a visible vertex.
    std::vector<int64_t> hash;

    /// Triangles are used to store the final triangulation simplices.
    std::vector<int64_t> triangles;
    std::unordered_map<int64_t, int64_t> halfedges;

    std::vector<int64_t> tri;

    /// These vectors provide an access to the convex hull of the triangulation,
    /// by implementing a double-linked list of vertices.
    std::vector<int64_t> next;
    std::vector<int64_t> prev;

    const torch::Tensor m_center;
    const torch::Tensor& m_points;

    int64_t start;

    shull(int64_t n, const torch::Tensor& center, const torch::Tensor& points)
    : hash(),
      triangles(),
      halfedges(),
      tri(n, -1),
      next(n, -1),
      prev(n, -1),
      m_center(center),
      m_points(points),
      start(0)
    {
        auto hash_size = static_cast<int64_t>(std::llround(std::ceil(std::sqrt(n))));
        hash.resize(hash_size);
        std::fill(hash.begin(), hash.end(), -1);
    }

    /// Computes the hash key for the given 2-dimensional point.
    ///
    /// The key represents pseudo-angle from the center of the triangulation.
    int64_t
    hash_key(const torch::Tensor& p) const
    {
        const auto delta = p - m_center;
        const auto dx = delta[0].item<double>();
        const auto dy = delta[1].item<double>();

        const auto rad = dx / (std::abs(dx) + std::abs(dy));
        const auto angle = (dy > 0.0 ? 3.0 - rad : 1.0 + rad) / 4.0;

        const auto k = std::llround(std::floor(angle * static_cast<double>(hash.size())));
        return static_cast<std::int64_t>(k) % hash.size();
    }

    void
    insert_visible_edge(int64_t i, int64_t j)
    {
        next[i] = j;
        prev[j] = i;

        hash[hash_key(m_points[i])] = i;
    }

    int64_t
    find_visible_edge(int64_t i) const
    {
        const auto key = hash_key(m_points[i]);
        int64_t edge_index = 0;

        for (int64_t j = 0; j < hash.size(); j++) {
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
    bool
    ccw(int64_t i0, int64_t i1, int64_t i2) const
    {
        const auto [p0, p1, p2] = get_tri(i0, i1, i2);
        return orient2d(p0, p1, p2).gt(0).all().item<bool>();
    }

    /// Returns true when specified points are ordered counterclockwise.
    inline bool
    ccw(const triangle_type& tri) const
    {
        return ccw(std::get<0>(tri), std::get<1>(tri), std::get<2>(tri));
    }

    /// Returns coordinates of the specified triangle.
    ///
    /// Method does not checks the boundaries of the specified indices, therefore, when
    /// indices are outside of the points bounds, the behaviour is undefined.
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    get_tri(int64_t i0, int64_t i1, int64_t i2) const
    {
        return std::forward_as_tuple(m_points[i0], m_points[i1], m_points[i2]);
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

    void
    push_edge(int64_t a, int64_t b)
    {
        if (a != -1) {
            halfedges[a] = b;
        }
        if (b != -1) {
            halfedges[b] = a;
        }
    }

    triangle_type
    forward(const triangle_type& triangle, int64_t edge)
    {
        auto [j, i, k] = triangle;
        tri[i] = edge;

        std::swap(j, next[j]);
        return std::forward_as_tuple(j, i, next[j]);
    }

    triangle_type
    backward(const triangle_type& triangle, int64_t edge)
    {
        auto [j, i, k] = triangle;
        tri[j] = edge;
        next[k] = k;

        std::swap(k, prev[k]);
        return std::forward_as_tuple(prev[k], i, k);
    }

    inline triangle_type
    tri_edges(int64_t edge) const
    {
        int64_t edge0 = 3 * (edge / 3);
        int64_t edge1 = edge0 + (edge + 1) % 3;
        int64_t edge2 = edge0 + (edge + 2) % 3;
        return std::forward_as_tuple(edge, edge1, edge2);
    }

    /// Flip triangle edges recursively until they all comply to Delaunay condition.
    ///
    /// \return The next edge of the triangle (in ccw direction).
    std::size_t
    flip(int64_t edge)
    {
        int64_t ar = -1, al = -1;
        int64_t br = -1, bl = -1;

        std::stack<int64_t> unvisited_edges({edge});

        while (unvisited_edges.size() > 0) {
            auto a = unvisited_edges.top();
            auto b = halfedges[a];

            std::tie(a, al, ar) = tri_edges(a);
            std::tie(b, br, bl) = tri_edges(b);

            unvisited_edges.pop();
            if (b == -1) {
                continue;
            }

            auto triangle = torch::tensor({triangles[ar], triangles[a], triangles[al]});
            auto p1 = m_points[triangles[bl]];

            if (incircle2d(m_points.index({triangle}), p1)) {
                triangles[a] = triangles[bl];
                triangles[b] = triangles[ar];

                if (halfedges[bl] == -1) {
                    auto ie = start;
                    do {
                        if (tri[ie] == bl) {
                            tri[ie] = a;
                            break;
                        }

                        ie = prev[ie];
                    } while (ie != start);
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


/// Compute seed triangle for the sweep-hull triangulation algorithm.
///
/// The method computes the center of all specified points and returns coordinates and indices
/// of the triangle comprised of points with the lowest distance from the center of point cloud.
std::tuple<torch::Tensor, torch::Tensor>
shull_seed2d(const torch::Tensor& points)
{
    // Indices of the seed triangle.
    torch::Tensor index0, index1, index2;
    torch::Tensor values, indices;

    // Choose seed points close to a centroid of the point cloud.
    auto [min, max] = points.aminmax(0);
    const auto centroid = (max + min) / 2;
    const auto dists = euclidean_distance2d(points, centroid);

    std::tie(values, indices) = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    index0 = indices[0];
    index1 = indices[1];

    auto p0 = points.index({index0});
    auto p1 = points.index({index1});

    // Find the third point such that forms the smallest circumcircle with i0 and i1.
    const auto radii = circumradius2d(p0.view({-1, 2}), p1.view({-1, 2}), points);

    std::tie(values, indices) = at::topk(radii, 3, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    // For points p0 and p1, radii of circumscribed circle will be set to `nan`, therefore
    // at 0 index will be a point with the minimum radius.
    index2 = indices[0];
    auto p2 = points.index({index2});

    if (ccw(p0, p1, p2)) {
        std::swap(index1, index2);
        std::swap(p1, p2);
    }

    auto points_out = at::vstack({p0, p1, p2});
    auto indices_out = at::vstack({index0, index1, index2});

    return std::forward_as_tuple(points_out, indices_out);
}


torch::Tensor
shull2d(const torch::Tensor& points)
{
    TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got: ", points.dim(), "D");

    // Find the seed triangle (comprised of vertices located approximately
    // at the center of the point cloud).
    const auto [triangle, indices] = shull_seed2d(points);

    // Radially resort the points.
    const auto center = circumcenter2d(triangle);
    const auto ordering = at::argsort(euclidean_distance2d(points, center));

    const auto n = points.size(0);
    shull hull(n, center, points);

    std::cout << "hull created" << std::endl;

    auto i0 = indices[0].item<int64_t>();
    auto i1 = indices[1].item<int64_t>();
    auto i2 = indices[2].item<int64_t>();

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
        auto i = ordering[k].item<int64_t>();
        std::cout << "processing [" << k << "]" << std::endl;

        auto is = hull.find_visible_edge(i);
        // TODO: Make sure what we found is on the hull?
        auto ie = hull.prev[is];

        while (hull.ccw(ie, i, hull.next[ie]) && ie != -1) {
            ie = hull.next[ie];
            std::cout << "\tfind place" << std::endl;
        }

        if (ie == -1) {
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
            std::cout << "\tforward" << std::endl;
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
                std::cout << "\tbackward" << std::endl;
            }
        }

        hull.start = ie;
        hull.insert_visible_edge(i, in);
        hull.insert_visible_edge(ie, i);
    }

    int64_t tn = hull.triangles.size() / 3;
    auto options = points.options().dtype(torch::kInt64);
    auto triangles = torch::tensor(std::move(hull.triangles), options);

    return triangles.view({tn, 3});
}


} // namespace torch_delaunay
