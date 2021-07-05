// Copyright 2020 the Polygon Mesh Processing Library developers.
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "Viewer.h"

int main(int argc, char** argv)
{
    Viewer window("Polygon Laplacian", 800, 600);

    if (argc == 2)
        window.load_mesh(argv[1]);

    return window.run();
}
