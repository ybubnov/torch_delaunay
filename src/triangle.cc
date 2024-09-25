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


void
_lawson_flip_out(
    const torch::Tensor& triangles,
    const torch::Tensor& halfedges,
    const torch::Tensor& points,
    const torch::Tensor& pr,
    const torch::Tensor& pl,
    const torch::Tensor& p0,
    torch::Tensor& out
)
{
    auto index_twin = halfedges.index_select(0, pl).index_select(1, pr);
    std::cout << "OPPOSITE" << std::endl << index_twin.to_dense() << std::endl;

    // TODO: implement diag for sparse coordinate tensor.
    auto twins = index_twin.to_dense().diag();
    twins = twins - 1;
    std::cout << "TWINS" << std::endl << twins << std::endl;

    twins = torch::where(twins.ne(-1), twins, p0);
    std::cout << "TWINS (defaulted)" << std::endl << twins << std::endl;

    // torch::Tensor p1 = triangles.index({twins - 1, dim});
    const auto p1 = twins;
    std::cout << "P1: " << p1 << std::endl;

    const auto incircle = incircle2d(
        points.index({p0}), points.index({pr}), points.index({pl}), points.index({p1})
    );
    // std::cout << "IN CIRCLE" << std::endl << incircle << std::endl;
    std::cout << "PP: " << torch::column_stack({p0, pr, pl, p1, incircle}) << std::endl;

    for (int64_t i = 0; i < out.sizes()[0]; i++) {
        if ((incircle[i].item<int64_t>() > 0)) {
            std::cout << "flip " << i << std::endl;
            out.index_put_({i, 0}, p0[i].item<int64_t>());
            out.index_put_({i, 1}, pr[i].item<int64_t>());
            out.index_put_({i, 2}, p1[i].item<int64_t>());
        }
    }
}


torch::Tensor
lawson_flip(const torch::Tensor& triangles, const torch::Tensor& points)
{
    torch::Tensor out = triangles.clone();

    const auto n = triangles.size(0);

    torch::Tensor he_indices = torch::full({2, n * 3}, -1, triangles.options());
    torch::Tensor he_values = torch::full({n * 3}, -1, triangles.options());

    for (const auto i : c10::irange(n)) {
        auto v0 = triangles.index({i, 0}).item<int64_t>();
        auto v1 = triangles.index({i, 1}).item<int64_t>();
        auto v2 = triangles.index({i, 2}).item<int64_t>();

        std::cout << i;
        he_indices.index_put_({0, i * 3 + 0}, v0);
        he_indices.index_put_({1, i * 3 + 0}, v1);

        he_indices.index_put_({0, i * 3 + 1}, v1);
        he_indices.index_put_({1, i * 3 + 1}, v2);

        he_indices.index_put_({0, i * 3 + 2}, v2);
        he_indices.index_put_({1, i * 3 + 2}, v0);
        std::cout << " ind... ";

        he_values.index_put_({i * 3 + 0}, v2 + 1);
        he_values.index_put_({i * 3 + 1}, v0 + 1);
        he_values.index_put_({i * 3 + 2}, v1 + 1);
        std::cout << "vals..." << std::endl;
    }

    const auto halfedges = torch::sparse_coo_tensor(he_indices, he_values);
    std::cout << "REV INDEX" << std::endl << halfedges.to_dense() << std::endl;

    torch::Tensor pr = triangles.index({Slice(), 0}).contiguous();
    torch::Tensor pl = triangles.index({Slice(), 1}).contiguous();
    torch::Tensor p0 = triangles.index({Slice(), 2}).contiguous();

    _lawson_flip_out(triangles, halfedges, points, pr, pl, p0, out);
    std::cout << "OUT" << std::endl << out << std::endl;
    _lawson_flip_out(triangles, halfedges, points, p0, pr, pl, out);
    std::cout << "OUT" << std::endl << out << std::endl;
    _lawson_flip_out(triangles, halfedges, points, pl, p0, pr, out);
    std::cout << "OUT" << std::endl << out << std::endl;

    return out;
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

    _SHull(int64_t n, double x, double y)
    : hash(),
      triangles(),
      halfedges(),
      tri(n, -1),
      hash_size(),
      next(n, -1),
      prev(n, -1),
      center_x(),
      center_y(),
      start(0)
    {
        hash_size = static_cast<int64_t>(std::llround(std::ceil(std::sqrt(n))));
        hash.resize(hash_size);
        std::fill(hash.begin(), hash.end(), -1);

        center_x = x;
        center_y = y;
    }

    std::size_t
    push_tri(int64_t i0, int64_t i1, int64_t i2, int64_t a, int64_t b, int64_t c)
    {
        auto t = triangles.size();
        triangles.push_back(i0);
        triangles.push_back(i1);
        triangles.push_back(i2);
        link(t, a);
        link(t + 1, b);
        link(t + 2, c);
        return t;
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

    std::size_t
    flip(int64_t a, const torch::Tensor& points)
    {
        int64_t i = 0;
        int64_t ar = 0;

        // TODO: is it possible to replace this with a list<int64_t>?
        std::vector<int64_t> edge_stack;
        while (true) {
            auto b = halfedges[a];

            auto a0 = 3 * (a / 3);
            ar = a0 + (a + 2) % 3;

            if (b == -1) {
                if (i > 0) {
                    i--;
                    a = edge_stack[i];
                    continue;
                } else {
                    break;
                }
            }

            auto b0 = 3 * (b / 3);
            auto al = a0 + (a + 1) % 3;
            auto bl = b0 + (b + 2) % 3;

            auto p0 = points[triangles[ar]].unsqueeze(0);
            auto pr = points[triangles[a]].unsqueeze(0);
            auto pl = points[triangles[al]].unsqueeze(0);
            auto p1 = points[triangles[bl]].unsqueeze(0);

            // TODO: is it >0 or <0?
            if (incircle2d(p0, pr, pl, p1).gt(0).all().item<bool>()) {
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

                if (i < edge_stack.size()) {
                    edge_stack[i] = br;
                } else {
                    edge_stack.push_back(br);
                }

                i++;
            } else {
                if (i > 0) {
                    i--;
                    a = edge_stack[i];
                    continue;
                } else {
                    break;
                }
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

    _SHull hull(n, center.index({0, 0}).item<float>(), center.index({0, 1}).item<float>());

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

        // TODO: correct up to this point.
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

        // TODO: triangles.push_back is missing method `link`.
        auto it = hull.push_tri(ie, i, hull.next[ie], -1, -1, hull.tri[ie]);
        hull.tri[i] = hull.flip(it + 2, points);
        hull.tri[ie] = it;


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

            auto it = hull.push_tri(in, i, iq, hull.tri[i], -1, hull.tri[in]);
            hull.tri[i] = hull.flip(it + 2, points);

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

                auto it = hull.push_tri(iq, i, ie, -1, hull.tri[ie], hull.tri[iq]);
                hull.flip(it + 2, points);
                hull.tri[iq] = it;

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

    const auto tn = static_cast<int64_t>(hull.triangles.size() / 3);
    const auto opts = torch::dtype(torch::kInt64);
    return at::from_blob(hull.triangles.data(), {tn, 3}, opts);
}


} // namespace torch_delaunay
