#pragma once

namespace il {

enum class HMatrixType { LowRank, FullRank, Hierarchical };

struct SubHMatrix {
  il::Range range0;
  il::Range range1;
  il::HMatrixType type;
};

}
