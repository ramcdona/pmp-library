// Copyright 2021 the Polygon Mesh Processing Library developers.
// Copyright 2021 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under a MIT-style license, see LICENSE.txt for details.

Eigen::Vector3d gradient_hat_function(Eigen::Vector3d i, Eigen::Vector3d j,
                                      Eigen::Vector3d k)
{
    const double eps = 1e-10;

    Eigen::Vector3d gradient, side, base, grad;
    double area = 0.5 * ((j - i).cross(k - i)).norm();
    side = i - j;
    base = k - j;
    grad = side - (side.dot(base) / base.norm()) * base / base.norm();
    if (area < eps)
    {
        gradient = Eigen::Vector3d(0, 0, 0);
    }
    else
    {
        grad = base.norm() * grad / grad.norm();
        gradient = grad / (2.0 * area);
    }

    return gradient;
}

void localGradientMatrix(const Eigen::MatrixXd& poly,
                         const Eigen::Vector3d& min, Eigen::VectorXd& w,
                         Eigen::MatrixXd& G)
{
    const int n = (int)poly.rows();

    G.resize(3 * n, n + 1);
    G.setZero();

    Eigen::Vector3d gradient_p, gradient_p0, gradient_p1, p, p0, p1;

    p = min;
    for (int i = 0; i < n; ++i)
    {
        const int i1 = (i + 1) % n;

        p0 = poly.row(i);
        p1 = poly.row(i1);

        gradient_p = gradient_hat_function(p, p0, p1);
        gradient_p0 = gradient_hat_function(p0, p1, p);
        gradient_p1 = gradient_hat_function(p1, p, p0);
        for (int j = 0; j < 3; j++)
        {
            G(3 * i + j, n) = gradient_p(j);
            G(3 * i + j, i) = gradient_p0(j);
            G(3 * i + j, i1) = gradient_p1(j);
        }
    }
}

void setup_sandwich_gradient_matrix(SurfaceMesh& mesh,
                                    Eigen::SparseMatrix<double>& G, int Laplace,
                                    int minpoint)
{
    const int nv = mesh.n_vertices();
    const int nf = mesh.n_faces();

    Eigen::MatrixXd Gi;
    Eigen::Vector3d min;
    Eigen::VectorXd w;
    Eigen::Vector3d p;
    Eigen::MatrixXd poly;

    std::vector<Eigen::Triplet<double>> trip;
    int nr_triangles = 0;
    int s = 0;
    for (Face f : mesh.faces())
    {
        const int n = mesh.valence(f);
        nr_triangles += n;
        poly.resize(n, 3);
        int i = 0;

        for (auto h : mesh.halfedges(f))
        {
            Vertex v = mesh.from_vertex(h);
            for (int h = 0; h < 3; h++)
            {
                poly.row(i)(h) = mesh.position(v)[h];
            }
            i++;
        }
        // compute weights for the polygon
        if (minpoint == Centroid)
        {
            int val = poly.rows();
            w = Eigen::MatrixXd::Ones(val, 1);
            w /= (double)val;
        }
        else if (minpoint == AbsAreaMinimizer)
        {
            Eigen::Vector3d point;
            optimizeAbsoluteTriangleArea(poly, point);
            find_weights_for_point(poly, point, w);
        }
        else
        {
            find_weights_fast(poly, w);
        }
        Eigen::Vector3d min;

        min = poly.transpose() * w;
        localGradientMatrix(poly, min, w, Gi);

        // sandwich
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < 3; k++)
                    Gi(3 * i + k, j) += w(j) * Gi(3 * i + k, n);

        int j = 0;
        int k;
        for (auto vv : mesh.vertices(f))
        {
            k = 0;
            for (auto h : mesh.halfedges(f))
            {
                Vertex v = mesh.from_vertex(h);
                for (int i = 0; i < 3; i++)
                {
                    trip.emplace_back(3 * s + i, v.idx(), Gi(3 * j + i, k));
                }
                k++;
            }
            j++;
            s++;
        }
    }

    G.resize(3 * nr_triangles, nv);
    G.setFromTriplets(trip.begin(), trip.end());
}

void setup_sandwich_divergence_matrix(SurfaceMesh& mesh,
                                      Eigen::SparseMatrix<double>& D,
                                      int Laplace, int minpoint)
{
    const int nv = mesh.n_vertices();
    const int nf = mesh.n_faces();

    Eigen::MatrixXd Gi, Di;
    Eigen::Vector3d min;
    Eigen::VectorXd w;
    Eigen::Vector3d p;
    Eigen::MatrixXd poly;

    std::vector<Eigen::Triplet<double>> trip;
    int nr_triangles = 0;
    int s = 0;
    for (Face f : mesh.faces())
    {
        const int n = mesh.valence(f);
        nr_triangles += n;
        poly.resize(n, 3);
        int i = 0;

        for (auto h : mesh.halfedges(f))
        {
            Vertex v = mesh.from_vertex(h);
            for (int h = 0; h < 3; h++)
            {
                poly.row(i)(h) = mesh.position(v)[h];
            }
            i++;
        }

        // compute weights for the polygon
        if (minpoint == Centroid)
        {
            int val = poly.rows();
            w = Eigen::MatrixXd::Ones(val, 1);
            w /= (double)val;
        }
        else if (minpoint == AbsAreaMinimizer)
        {
            Eigen::Vector3d point;
            optimizeAbsoluteTriangleArea(poly, point);
            find_weights_for_point(poly, point, w);
        }
        else
        {
            find_weights_fast(poly, w);
        }

        Eigen::Vector3d min = poly.transpose() * w;
        localGradientMatrix(poly, min, w, Gi);
        Di = -Gi.transpose();

        // triangle area diagonal matrix
        Eigen::MatrixXd Ai;
        Ai.resize(3 * n, 3 * n);
        Ai.setZero();
        for (int i = 0; i < n; ++i)
        {
            const int i1 = (i + 1) % n;

            Eigen::Vector3d p0 = poly.row(i);
            Eigen::Vector3d p1 = poly.row(i1);

            double area = 0.5 * ((p0 - min).cross(p1 - min)).norm();
            for (int k = 0; k < 3; k++)
            {
                Ai(3 * i + k, 3 * i + k) = area;
            }
        }
        Di *= Ai;

        // sandwich
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int k = 0; k < 3; k++)
                    Di(i, 3 * j + k) += w(i) * Di(n, 3 * j + k);
            }
        }

        int j = 0;
        int k;
        for (auto vv : mesh.vertices(f))
        {
            k = 0;
            for (auto h : mesh.halfedges(f))
            {
                Vertex v = mesh.from_vertex(h);
                for (int i = 0; i < 3; i++)
                {
                    trip.emplace_back(v.idx(), 3 * s + i, Di(k, 3 * j + i));
                }
                k++;
            }
            j++;
            s++;
        }
    }

    D.resize(nv, 3 * nr_triangles);
    D.setFromTriplets(trip.begin(), trip.end());
}
