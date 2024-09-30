#include <stack>
#include <unordered_map>

#include <torch_delaunay/predicates.h>
#include <torch_delaunay/triangle.h>

#include <ATen/native/cpu/Loops.h>
#include <torch/linalg.h>


using namespace torch::indexing;


namespace torch_delaunay {


std::tuple<torch::Tensor, torch::Tensor>
_cc_coordinates(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    auto a = p0;
    auto b = -p0 + p1;
    auto c = -p0 + p2;

    std::initializer_list<at::indexing::TensorIndex> x = {Slice(), 0};
    std::initializer_list<at::indexing::TensorIndex> y = {Slice(), 1};

    auto d = 2 * (b.index(x) * c.index(y) - b.index(y) * c.index(x));

    auto b_norm = b.square().sum({1});
    auto c_norm = c.square().sum({1});

    auto ux = (c.index(y) * b_norm - b.index(y) * c_norm) / d;
    auto uy = (b.index(x) * c_norm - c.index(x) * b_norm) / d;

    return std::forward_as_tuple(ux, uy);
}


torch::Tensor
circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    auto [ux, uy] = _cc_coordinates(p0, p1, p2);
    return at::column_stack({ux, uy}) + p0;
}


torch::Tensor
circumcenter(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    auto center = circumcenter2d(p0.unsqueeze(0), p1.unsqueeze(0), p2.unsqueeze(0));
    return center.squeeze(0);
}


// TODO: change circumradius implementation, so it does not produce nans and returns 0 instead.
torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    TORCH_CHECK(p0.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p0.dim(), "D");
    TORCH_CHECK(p1.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p1.dim(), "D");
    TORCH_CHECK(p2.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p2.dim(), "D");

    auto [ux, uy] = _cc_coordinates(p0, p1, p2);
    return (ux.square() + uy.square()).sqrt();
}


torch::Tensor
dist(const torch::Tensor& p, const torch::Tensor& q)
{
    TORCH_CHECK(p.dim() == 2, "dist only supports 2D tensors, got: ", p.dim(), "D");

    return torch::linalg::norm(p - q, /*ord=*/2, /*dim=*/1, false, c10::nullopt);
}


struct _SHull {
    std::vector<int64_t> hash;
    std::vector<int64_t> triangles;
    std::unordered_map<int64_t, int64_t> halfedges;
    std::vector<int64_t> tri;
    std::int64_t hash_size;

    std::vector<int64_t> next;
    std::vector<int64_t> prev;

    const torch::Tensor m_center;
    int64_t start;

    const torch::Tensor& m_points;

    _SHull(int64_t n, const torch::Tensor& center, const torch::Tensor& points)
    : hash(),
      triangles(),
      halfedges(),
      tri(n, -1),
      hash_size(),
      next(n, -1),
      prev(n, -1),
      m_center(center),
      start(0),
      m_points(points)
    {
        hash_size = static_cast<int64_t>(std::llround(std::ceil(std::sqrt(n))));
        hash.resize(hash_size);
        std::fill(hash.begin(), hash.end(), -1);
    }

    int64_t
    hash_key(const torch::Tensor& p) const
    {
        const auto delta = p - m_center;
        const auto dx = delta[0].item<double>();
        const auto dy = delta[1].item<double>();

        // pseudo angle
        const auto rad = dx / (std::abs(dx) + std::abs(dy));
        const auto angle = (dy > 0.0 ? 3.0 - rad : 1.0 + rad) / 4.0;

        const auto k = std::llround(std::floor(angle * static_cast<double>(hash_size)));
        return static_cast<std::int64_t>(k) % hash_size;
    }

    int64_t
    find_visible_edge(const torch::Tensor& point) const
    {
        const auto key = hash_key(point);
        int64_t edge_index = 0;

        for (int64_t j = 0; j < hash_size; j++) {
            edge_index = hash[(key + j) % hash_size];

            // TODO: why edge_index could be equal to next[edge_index]?
            if (edge_index != -1 && edge_index != next[edge_index]) {
                break;
            }
        }

        return edge_index;
    }

    void
    set(int64_t key, int64_t val)
    {
        hash[key] = val;
    }

    std::tuple<int64_t, int64_t>
    push_tri(int64_t i0, int64_t i1, int64_t i2, int64_t a, int64_t b, int64_t c)
    {
        auto edge = triangles.size();

        triangles.push_back(i0);
        triangles.push_back(i1);
        triangles.push_back(i2);
        push_edge(edge, a);
        push_edge(edge + 1, b);
        push_edge(edge + 2, c);

        // TODO: what does `x` mean?
        auto x = flip(edge + 2);
        return std::forward_as_tuple(edge, x);
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

    inline std::tuple<int64_t, int64_t, int64_t>
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


inline bool
ccw(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    return orient2d(p0, p1, p2).gt(0).all().item<bool>();
}


torch::Tensor
shull2d(const torch::Tensor& points)
{
    TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got: ", points.dim(), "D");

    // Indices of the seed triangle.
    torch::Tensor index0, index1, index2;
    const auto n = points.size(0);

    // Choose seed points close to a centroid of the point cloud.
    {
        auto [min, max] = points.aminmax(0);

        const auto centroid = (max + min) / 2;
        const auto dists = torch_delaunay::dist(points, centroid);

        auto [values, indices] = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);

        index0 = indices[0];
        index1 = indices[1];
    }

    auto p0 = points.index({index0});
    auto p1 = points.index({index1});

    // Find the third point such that forms the smallest circumcircle with i0 and i1.
    {
        // TODO: repeat increases the space, consider using broadcast operations.
        const auto radii = circumradius2d(p0.repeat({n, 1}), p1.repeat({n, 1}), points);

        auto [values, indices] = at::topk(radii, 3, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);

        // For points p0 and p1, radii of circumscribed circle will be set to `nan`, therefore
        // at 0 index will be a point with the minimum radius.
        index2 = indices[0];
    }

    auto p2 = points.index({index2});

    if (ccw(p0, p1, p2)) {
        std::cout << "swapped" << std::endl;
        std::swap(index1, index2);
        std::swap(p1, p2);
    }

    std::cout << "seed triangle chosen" << std::endl;

    // Radially resort the points.
    const auto center = circumcenter(p0, p1, p2);
    const auto ordering = at::argsort(torch_delaunay::dist(points, center));

    _SHull hull(n, center, points);

    std::cout << "hull created" << std::endl;

    auto i0 = index0.item<int64_t>();
    auto i1 = index1.item<int64_t>();
    auto i2 = index2.item<int64_t>();

    std::cout << "indices initialized" << std::endl;

    hull.start = i0;

    hull.set(hull.hash_key(p0), i0);
    hull.set(hull.hash_key(p1), i1);
    hull.set(hull.hash_key(p2), i2);

    hull.next[i0] = i1;
    hull.next[i1] = i2;
    hull.next[i2] = i0;

    hull.prev[i0] = i2;
    hull.prev[i1] = i0;
    hull.prev[i2] = i1;

    hull.tri[i0] = 0;
    hull.tri[i1] = 1;
    hull.tri[i2] = 2;

    std::cout << "seed: " << i0 << "," << i1 << "," << i2 << std::endl;
    hull.push_tri(i0, i1, i2, -1, -1, -1);

    for (const auto k : c10::irange(n)) {
        auto i = ordering[k].item<int64_t>();
        std::cout << "processing [" << k << "]" << std::endl;

        // Skip points of the seed triangle, they are already part of the final hull.
        if (i == i0 || i == i1 || i == i2) {
            continue;
        }

        int64_t is = hull.find_visible_edge(points[i]);

        // TODO: Make sure what we found is on the hull?

        is = hull.prev[is];
        int64_t ie = is;
        int64_t iq = is;

        // Advance until we find a place in the hull were the current point can be added.
        do {
            ie = iq;
            iq = hull.next[ie];
        } while (!ccw(points[i], points[ie], points[iq]));

        // TODO: Likely a near-duplicate?
        assert(ie != -1);

        auto [first_tri, last_tri] = hull.push_tri(ie, i, hull.next[ie], -1, -1, hull.tri[ie]);
        hull.tri[i] = last_tri;
        hull.tri[ie] = first_tri;

        // Traverse forward through the hull, adding more triangles and flipping
        // them recursively.
        auto in = hull.next[ie];

        while (true) {
            iq = hull.next[in];

            if (!ccw(points[i], points[in], points[iq])) {
                break;
            }

            auto [_, last_tri] = hull.push_tri(in, i, iq, hull.tri[i], -1, hull.tri[in]);
            hull.tri[i] = last_tri;

            hull.next[in] = in;
            in = iq;
        }

        // Traverse backward through the hull, adding more triangles and flipping
        // them recursively.
        if (ie == is) {
            iq = is;

            while (true) {
                iq = hull.prev[ie];

                if (!ccw(points[i], points[iq], points[ie])) {
                    break;
                }

                auto [first_tri, _] = hull.push_tri(iq, i, ie, -1, hull.tri[ie], hull.tri[iq]);
                hull.tri[iq] = first_tri;

                hull.next[ie] = ie;
                ie = iq;
            }
        }

        hull.start = ie;
        hull.prev[i] = ie;
        hull.prev[in] = i;
        hull.next[ie] = i;
        hull.next[i] = in;

        hull.set(hull.hash_key(points[i]), i);
        hull.set(hull.hash_key(points[ie]), ie);
    }

    std::cout << "triangles computed" << std::endl;

    int64_t tn = hull.triangles.size() / 3;
    auto answer
        = torch::tensor(std::move(hull.triangles), torch::TensorOptions().dtype(torch::kInt64));
    std::cout << "triangles move to tensor" << std::endl;
    return answer.view({tn, 3});
}


} // namespace torch_delaunay
