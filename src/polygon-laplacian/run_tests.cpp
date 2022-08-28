// Copyright 2020 the Polygon Mesh Processing Library developers.
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include "pmp/algorithms/PolyDiffGeo.h"
#include <pmp/algorithms/Subdivision.h>
#include <pmp/Timer.h>

#include "MeanCurvature.h"
#include "GeodesicsInHeat.h"

using namespace pmp;

// normalize all vertex position to norm=1.
// used to turn an almost spherical mesh into a spherical mesh.
void normalize(SurfaceMesh& mesh)
{
    for (auto v : mesh.vertices())
    {
        mesh.position(v) = normalize(mesh.position(v));
    }
}

// compute curvature on a spherical mesh.
// compare to ground truth, i.e., 1
void test_curvatures(SurfaceMesh& mesh)
{
    Curvature curvature(mesh);
    curvature.compute();

    double rms = 0.0;
    for (auto v : mesh.vertices())
    {
        double c = curvature(v);
        rms += (c - 1.0) * (c - 1.0);
    }
    rms /= (double)mesh.n_vertices();
    rms = sqrt(rms);

    std::cout << "rms error = " << rms << std::endl;
}

// compute geodesic distances from vertex 0 on a planar mesh.
// compare to ground truth, i.e., Euclidean distance.
void test_geodesics(SurfaceMesh& mesh)
{
    Vertex source(0);
    const Point p = mesh.position(source);

    GeodesicsInHeat heat(mesh);
    heat.compute_distance_from(source);

    double rms = 0.0;
    for (auto v : mesh.vertices())
    {
        double geodesic_dist = heat(v);
        double euclidean_dist = distance(p, mesh.position(v));
        rms += pow(geodesic_dist - euclidean_dist, 2.0);
    }
    rms /= (double)mesh.n_vertices();
    rms = sqrt(rms);

    std::cout << "rms error = " << rms << std::endl;
}

// measure time for constructing stiffness matrix
void timing_construction(SurfaceMesh mesh)
{
    const int trials = 20;

    pmp::Timer t;
    t.start();

    Eigen::SparseMatrix<double> S;
    for (int i = 0; i < trials; i++)
    {
        setup_stiffness_matrix(mesh, S);
    }

    t.stop();
    std::cout << "stiffness matrix construction: " << t.elapsed() / trials
              << "ms\n";
}

// measure time for factoring/solving Poisson system
void timing_solution(SurfaceMesh mesh)
{
    auto points = mesh.vertex_property<Point>("v:point");

    // setup stiffness matrix
    Eigen::SparseMatrix<double> S;
    setup_stiffness_matrix(mesh, S);

    // setup mass matrix
    Eigen::SparseMatrix<double> M;
    setup_mass_matrix(mesh, M);

    // setup rhs
    Eigen::MatrixXd B(mesh.n_vertices(), 3);
    for (auto v : mesh.vertices())
    {
        B.row(v.idx()) = static_cast<Eigen::Vector3d>(points[v]);
    }

    // setup implicit smoothing system
    Eigen::SparseMatrix<double> A;
    A = M - 0.1 * S;
    Eigen::MatrixXd M_B = M * B;

    static Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    const int trials = 10;

    // time for factoring matrix
    pmp::Timer t;
    t.start();
    for (int i = 0; i < trials; i++)
    {
        solver.compute(A);
    }
    t.stop();
    std::cout << "factorize matrix: " << t.elapsed() / trials << "ms\n";

    // time for back-substitution
    t.start();
    for (int i = 0; i < trials; i++)
    {
        Eigen::MatrixXd X = solver.solve(M_B);
    }
    t.stop();
    std::cout << "back-substitution: " << t.elapsed() / trials << "ms\n";
}

int main(int argc, char** argv)
{
    SurfaceMesh mesh;

    // which test to run
    int mytest = (argc > 1) ? atoi(argv[1]) : 0;

    // counter for tests
    int test = 1;

    // compute mean curvature
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature: Hex sphere\n";
            mesh.read("../data/unit-sphere.off");
            dual(mesh);
            normalize(mesh);
            test_curvatures(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature: Fine sphere\n";
            mesh.read("../data/unit-sphere.off");
            Subdivision(mesh).catmull_clark();
            normalize(mesh);
            test_curvatures(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature: Regular sphere\n";
            mesh.read("../data/quad-sphere.off");
            normalize(mesh);
            test_curvatures(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature : Noisy sphere\n";
            mesh.read("../data/noisy-sphere.off");
            normalize(mesh);
            test_curvatures(mesh);
        }
    }
    ++test;

    // compute geodesic distance
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: Subdivided quad plane \n";
            mesh.read("../data/quad-plane.obj");
            Subdivision(mesh).catmull_clark();
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: Quad plane\n";
            mesh.read("../data/quad-plane.obj");
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: L-plane\n";
            mesh.read("../data/L-plane.obj");
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: Tetris-plane (non-starshaped)\n";
            mesh.read("../data/tetris.obj");
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist:  Tetris-plane (non-convex)\n";
            mesh.read("../data/tetris_2.obj");
            test_geodesics(mesh);
        }
    }
    ++test;

    // measure timings
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Timings: Hexasphere\n";
            mesh.read("../data/Spheres/hexaSphere.off");
            timing_construction(mesh);
            timing_solution(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Timings: Fine Sphere\n";
            mesh.read("../data/Spheres/fineSphere.off");
            timing_construction(mesh);
            timing_solution(mesh);
        }
    }
}
