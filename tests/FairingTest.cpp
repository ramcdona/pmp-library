// Copyright 2017-2021 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "gtest/gtest.h"

#include "pmp/algorithms/Fairing.h"
#include "pmp/algorithms/Subdivision.h"
#include "pmp/utilities.h"
#include "Helpers.h"

using namespace pmp;

TEST(FairingTest, fairing)
{
    auto mesh = open_cone();
    Subdivision(mesh).loop();
    auto bbz = bounds(mesh).max()[2];
    Fairing sf(mesh);
    sf.fair();
    auto bbs = bounds(mesh).max()[2];
    EXPECT_LT(bbs, bbz);
}

TEST(FairingTest, fairing_selected)
{
    auto mesh = open_cone();
    Subdivision(mesh).loop();

    auto bb = bounds(mesh);

    // select top vertices for fairing
    auto selected = mesh.vertex_property<bool>("v:selected");
    for (auto v : mesh.vertices())
        if (mesh.position(v)[2] > 0.5)
            selected[v] = true;

    Fairing sf(mesh);
    sf.fair();
    auto bb2 = bounds(mesh);
    EXPECT_LT(bb2.size(), bb.size());
}
