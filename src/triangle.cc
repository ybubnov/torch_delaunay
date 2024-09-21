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

    // TODO: implement diag for sparse coo tensor.
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
    std::vector<torch::Tensor> hash;
    std::int64_t hash_size;

    torch::Tensor next;
    torch::Tensor prev;

    double center_x;
    double center_y;

    _SHull(int64_t n, double x, double y)
    : hash(),
      hash_size(),
      next(),
      prev(),
      center_x(),
      center_y()
    {
        hash_size = static_cast<int64_t>(std::llround(std::ceil(std::sqrt(n))));
        hash.resize(hash_size);
        std::fill(hash.begin(), hash.end(), torch::tensor(-1));

        next = torch::full({n}, -1, torch::dtype(torch::kInt64));
        prev = torch::full({n}, -1, torch::dtype(torch::kInt64));

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

    torch::Tensor
    get(const int64_t key) const
    {
        return hash[key % hash_size];
    }

    void
    set(const int64_t key, torch::Tensor val)
    {
        hash[key] = val;
    }
};


void
print_triangle(
    const char* name, const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2
)
{
    std::cout << name << ": " << p0.item<int64_t>() << " " << p1.item<int64_t>() << " "
              << p2.item<int64_t>() << std::endl;
}


torch::Tensor
shull2d(const torch::Tensor& points)
{
    TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got: ", points.dim(), "D");

    // Indices of the seed triangle.
    torch::Tensor i0, i1, i2;
    const auto n = points.size(0);

    // Choose seed points close to a centroid of the point cloud.
    {
        torch::Tensor min, max;
        std::tie(min, max) = points.aminmax(0);

        const auto centroid = (max + min) / 2;
        std::cout << "centroid: " << centroid << std::endl;
        const auto dists = torch_delaunay::dist(points, centroid);
        std::cout << "dists: " << dists << std::endl;

        torch::Tensor values, indices;
        std::tie(values, indices)
            = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=1*/ true);

        i0 = indices[0];
        i1 = indices[1];
    }

    std::cout << "i0: " << i0.item<int64_t>() << " i1: " << i1.item<int64_t>() << std::endl;

    auto p0 = points[i0].unsqueeze(0);
    auto p1 = points[i1].unsqueeze(0);

    // Find the third point such that forms the smallest circumcircle with i0 and i1.
    {
        // const auto radiuses = torch_delaunay::en::circumradius2d(points, p0, p1);
        const auto radiuses
            = circumradius2d(p0.repeat({n, 1}), p1.repeat({n, 1}), points);

        torch::Tensor values, indices;
        std::tie(values, indices)
            = at::topk(radiuses, 3, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
        std::cout << "radiuses: " << radiuses << std::endl;
        std::cout << "circumradius/indices: " << indices << std::endl;
        i2 = indices[0];
    }

    auto p2 = points[i2].unsqueeze(0);

    // i1 = torch::tensor(2, torch::dtype(torch::kInt64));
    // p1 = points[i1].unsqueeze(0);

    if (all_counterclockwise2d(p0, p1, p2)) {
        std::cout << "swapped" << std::endl;
        std::swap(i1, i2);
        std::swap(p1, p2);
    }

    std::cout << "seed triangle chosen" << std::endl;

    // Radially resort the points.
    const auto center = circumcenter2d(p0, p1, p2);

    // std::cout << "circumcenter: " << center << std::endl;
    // std::cout << "distances: " << torch_delaunay::en::dist(points, center) << std::endl;
    const auto ordering = at::argsort(torch_delaunay::dist(points, center));
    // std::cout << "ordering: " << ordering << std::endl;

    _SHull hull(n, center.index({0, 0}).item<float>(), center.index({0, 1}).item<float>());

    std::cout << "hull created" << std::endl;

    i0 = i0.clone();
    i1 = i1.clone();
    i2 = i2.clone();

    hull.set(hull.key(p0), i0);
    hull.set(hull.key(p1), i1);
    hull.set(hull.key(p2), i2);

    std::cout << "hash: " << hull.hash << std::endl;

    hull.next.index_put_({i0}, i1);
    hull.next.index_put_({i1}, i2);
    hull.next.index_put_({i2}, i0);

    hull.prev.index_put_({i0}, i2);
    hull.prev.index_put_({i1}, i0);
    hull.prev.index_put_({i2}, i1);

    std::cout << "HULL NEXT: \n" << hull.next << std::endl;
    std::cout << "HULL PREV: \n" << hull.prev << std::endl;
    std::vector<int64_t> faces;

    print_triangle("Ts", i0, i1, i2);
    faces.push_back(i0.item<int64_t>());
    faces.push_back(i2.item<int64_t>());
    faces.push_back(i1.item<int64_t>());

    for (const auto k : c10::irange(n)) {
        auto i = ordering.index({k});
        const auto pi = points.index({i}).unsqueeze(0);

        std::cout << "processing " << i.item<int64_t>() << " (" << pi[0][0].item<float>() << ","
                  << pi[0][1].item<float>() << ")" << std::endl;

        if ((i == i0).all().item<bool>() || (i == i1).all().item<bool>()
            || (i == i2).all().item<bool>()) {
            continue;
        }

        const auto key = hull.key(pi);

        torch::Tensor is = torch::tensor(0, torch::dtype(torch::kInt64));
        for (int64_t j = 0; j < hull.size(); j++) {
            is = hull.get(key + j);
            std::cout << "next key: " << (key + j) << std::endl;
            if (is.ne(-1).all().item<bool>() && is.ne(hull.next[is]).all().item<bool>()) {
                break;
            }
        }

        std::cout << "is: " << is.item<int64_t>() << " -> " << hull.prev[is].item<int64_t>()
                  << std::endl;
        is = hull.prev[is];
        auto ie = is;
        auto iq = is;

        torch::Tensor pe, pn, pq;
        while (true) {
            ie = iq;
            iq = hull.next[ie];

            pe = points[ie].unsqueeze(0);
            pq = points[iq].unsqueeze(0);
            print_triangle("  ti", i, ie, hull.next[ie]);

            if (all_counterclockwise2d(pi, pe, pq)) {
                break;
            }
        }

        faces.push_back(ie.item<int64_t>());
        faces.push_back(hull.next[ie].item<int64_t>());
        faces.push_back(i.item<int64_t>());

        print_triangle("Ti", ie, i, hull.next[ie]);

        // Traverse forward.
        auto in = hull.next[ie];

        while (true) {
            iq = hull.next[in].clone();

            pn = points[in].unsqueeze(0);
            pq = points[iq].unsqueeze(0);

            if (!all_counterclockwise2d(pi, pn, pq)) {
                break;
            }

            // Add a new triangle.
            print_triangle("Tn", in, i, iq);

            faces.push_back(in.item<int64_t>());
            faces.push_back(iq.item<int64_t>());
            faces.push_back(i.item<int64_t>());

            in = in.clone();
            hull.next.index_put_({in}, in);
            in = iq.clone();

            print_triangle(" tn", in, i, iq);
        }

        if ((ie == is).all().item<bool>()) {
            iq = is;

            while (true) {
                iq = hull.prev[ie].clone();

                pe = points[ie].unsqueeze(0);
                pq = points[iq].unsqueeze(0);

                if (!all_counterclockwise2d(pi, pq, pe)) {
                    break;
                }

                // Add a new triangle.
                print_triangle("Te", iq, i, ie);
                std::cout << "  te " << orient2d(pi, pq, pe)[0].item<int64_t>()
                          << std::endl;

                faces.push_back(iq.item<int64_t>());
                faces.push_back(ie.item<int64_t>());
                faces.push_back(i.item<int64_t>());

                ie = ie.clone();
                hull.next.index_put_({ie}, ie.clone());
                ie = iq.clone();
            }
        }

        i = i.clone();
        ie = ie.clone();
        in = in.clone();

        hull.prev.index_put_({i}, ie);
        hull.prev.index_put_({in}, i);
        hull.next.index_put_({ie}, i);
        hull.next.index_put_({i}, in);
        std::cout << "HULL (next): " << hull.next << std::endl;
        std::cout << "HULL (prev): " << hull.prev << std::endl;

        hull.set(key, i);
        hull.set(hull.key(points[ie].unsqueeze(0)), ie);
    }

    const auto tn = static_cast<int64_t>(faces.size() / 3);

    const auto opts = torch::dtype(torch::kInt64);
    return at::from_blob(faces.data(), {tn, 3}, opts).clone();
}


} // namespace torch_delaunay
