#include <stack>

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
    TORCH_CHECK(p0.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p0.dim(), "D");
    TORCH_CHECK(p1.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p1.dim(), "D");
    TORCH_CHECK(p2.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p2.dim(), "D");

    torch::Tensor ux, uy;
    std::tie(ux, uy) = _cc_coordinates(p0, p1, p2);

    return at::column_stack({ux, uy}) + p0;
}


torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    torch::Tensor ux, uy;
    std::tie(ux, uy) = _cc_coordinates(p0, p1, p2);

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
    std::vector<int64_t> halfedges;
    std::vector<int64_t> tri;
    std::int64_t hash_size;

    std::vector<int64_t> next;
    std::vector<int64_t> prev;

    double center_x;
    double center_y;
    int64_t start;

    const torch::Tensor& m_points;

    _SHull(int64_t n, double x, double y, const torch::Tensor& points)
    : hash(),
      triangles(),
      halfedges(),
      tri(n, -1),
      hash_size(),
      next(n, -1),
      prev(n, -1),
      center_x(),
      center_y(),
      start(0),
      m_points(points)
    {
        hash_size = static_cast<int64_t>(std::llround(std::ceil(std::sqrt(n))));
        hash.resize(hash_size);
        std::fill(hash.begin(), hash.end(), -1);

        center_x = x;
        center_y = y;
    }

    int64_t
    size() const
    {
        return hash_size;
    }

    int64_t
    key(const torch::Tensor& p) const
    {
        const auto dx = p[0][0].item<float>() - center_x;
        const auto dy = p[0][1].item<float>() - center_y;

        // pseudo angle
        const auto rad = dx / (std::abs(dx) + std::abs(dy));
        const auto angle = (dy > 0.0 ? 3.0 - rad : 1.0 + rad) / 4.0;

        const auto k = std::llround(std::floor(angle * static_cast<double>(hash_size)));
        return static_cast<std::int64_t>(k) % hash_size;
    }

    int64_t
    get(int64_t key) const
    {
        return hash[key % hash_size];
    }

    void
    set(int64_t key, int64_t val)
    {
        hash[key] = val;
    }

    std::tuple<int64_t, int64_t>
    push_tri(int64_t i0, int64_t i1, int64_t i2, int64_t a, int64_t b, int64_t c)
    {
        auto t = triangles.size();

        triangles.push_back(i0);
        triangles.push_back(i1);
        triangles.push_back(i2);
        link(t, a);
        link(t + 1, b);
        link(t + 2, c);

        // TODO: what does `x` mean?
        auto x = flip(t + 2);
        return std::make_tuple(t, x);
    }

    void
    link(int64_t a, int64_t b)
    {
        auto num_halfedges = halfedges.size();
        if (a == num_halfedges) {
            halfedges.push_back(b);
        } else if (a < num_halfedges) {
            halfedges[a] = b;
        } else {
            throw std::runtime_error("cannot link the edge");
        }

        if (b == -1) {
            return;
        }

        num_halfedges = halfedges.size();
        if (b == num_halfedges) {
            halfedges.push_back(a);
        } else if (b < num_halfedges) {
            halfedges[b] = a;
        } else {
            throw std::runtime_error("cannot link the edge");
        }
    }

    std::size_t
    flip(int64_t a)
    {
        int64_t ar = 0;
        std::stack<int64_t> edge_stack({a});

        while (edge_stack.size() > 0) {
            int64_t a = edge_stack.top();
            edge_stack.pop();

            auto a0 = 3 * (a / 3);
            ar = a0 + (a + 2) % 3;

            auto b = halfedges[a];
            if (b == -1) {
                continue;
            }

            auto b0 = 3 * (b / 3);
            auto al = a0 + (a + 1) % 3;
            auto bl = b0 + (b + 2) % 3;

            auto p0 = m_points[triangles[ar]].unsqueeze(0);
            auto pr = m_points[triangles[a]].unsqueeze(0);
            auto pl = m_points[triangles[al]].unsqueeze(0);
            auto p1 = m_points[triangles[bl]].unsqueeze(0);

            if (incircle2d(p0, pr, pl, p1).lt(0).all().item<bool>()) {
                triangles[a] = triangles[bl];
                triangles[b] = triangles[ar];

                auto h_bl = halfedges[bl];

                if (h_bl == -1) {
                    auto ie = start;
                    do {
                        if (tri[ie] == bl) {
                            tri[ie] = a;
                            break;
                        }

                        ie = prev[ie];
                    } while (ie != start);
                }

                link(a, h_bl);
                link(b, halfedges[ar]);
                link(ar, bl);

                auto br = b0 + (b + 1) % 3;

                edge_stack.push(br);
                edge_stack.push(a);
            }
        }

        return ar;
    }
};


torch::Tensor
shull2d(const torch::Tensor& points)
{
    TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got: ", points.dim(), "D");

    // Indices of the seed triangle.
    torch::Tensor index0, index1, index2;
    const auto n = points.size(0);
    std::cout << "points: " << points << std::endl;

    // Choose seed points close to a centroid of the point cloud.
    {
        auto [min, max] = points.aminmax(0);

        const auto centroid = (max + min) / 2;
        std::cout << "centroid: " << centroid << std::endl;
        const auto dists = torch_delaunay::dist(points, centroid);
        std::cout << "dists: " << dists << std::endl;

        auto [values, indices] = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);

        index0 = indices[0];
        index1 = indices[1];
    }

    auto p0 = points[index0].unsqueeze(0);
    auto p1 = points[index1].unsqueeze(0);

    // Find the third point such that forms the smallest circumcircle with i0 and i1.
    {
        // TODO: repeat increases the space, consider using broadcast operations.
        const auto radii = circumradius2d(p0.repeat({n, 1}), p1.repeat({n, 1}), points);

        auto [values, indices] = at::topk(radii, 3, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
        std::cout << "radii: " << radii << std::endl;
        std::cout << "circumradius/indices: " << indices << std::endl;

        // For points p0 and p1, radii of circumscribed circle will be set to `nan`, therefore
        // at 0 index will be a point with the minimum radius.
        index2 = indices[0];
    }

    auto p2 = points[index2].unsqueeze(0);

    if (all_counterclockwise2d(p0, p1, p2)) {
        std::cout << "swapped" << std::endl;
        std::swap(index1, index2);
        std::swap(p1, p2);
    }

    std::cout << "seed triangle chosen" << std::endl;

    // Radially resort the points.
    const auto center = circumcenter2d(p0, p1, p2);

    const auto ordering = at::argsort(torch_delaunay::dist(points, center));

    _SHull hull(n, center.index({0, 0}).item<float>(), center.index({0, 1}).item<float>(), points);

    std::cout << "hull created" << std::endl;

    auto i0 = index0.item<int64_t>();
    auto i1 = index1.item<int64_t>();
    auto i2 = index2.item<int64_t>();

    hull.start = i0;

    hull.set(hull.key(p0), i0);
    hull.set(hull.key(p1), i1);
    hull.set(hull.key(p2), i2);

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
        const auto pi = points[i].unsqueeze(0);

        // Skip points of the seed triangle, they are already part of the final hull.
        if (i == i0 || i == i1 || i == i2) {
            continue;
        }

        const auto key = hull.key(pi);

        int64_t is = 0;
        for (int64_t j = 0; j < hull.size(); j++) {
            is = hull.get(key + j);
            std::cout << "next key: " << (key + j) << std::endl;
            if (is != -1 && is != hull.next[is]) {
                break;
            }
        }

        // TODO: Make sure what we found is on the hull?

        is = hull.prev[is];
        int64_t ie = is;
        int64_t iq = is;

        // Advance until we find a place in the hull were the current point can be added.
        torch::Tensor pe, pn, pq;
        while (true) {
            ie = iq;
            iq = hull.next[ie];

            pe = points[ie].unsqueeze(0);
            pq = points[iq].unsqueeze(0);

            if (all_counterclockwise2d(pi, pe, pq)) {
                break;
            }
        }

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

            pn = points[in].unsqueeze(0);
            pq = points[iq].unsqueeze(0);

            if (!all_counterclockwise2d(pi, pn, pq)) {
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

                pq = points[iq].unsqueeze(0);
                pe = points[ie].unsqueeze(0);

                if (!all_counterclockwise2d(pi, pq, pe)) {
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

        hull.set(key, i);
        hull.set(hull.key(points[ie].unsqueeze(0)), ie);
    }

    int64_t tn = hull.triangles.size() / 3;
    auto answer = torch::tensor(hull.triangles, torch::TensorOptions().dtype(torch::kInt64));
    return answer.reshape({tn, 3});
}


} // namespace torch_delaunay
