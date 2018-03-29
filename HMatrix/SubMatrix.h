#pragma once

#include <il/core.h>
#include "HMatrixType.h"

namespace il {

struct SubMatrix {
  il::Range range0;
  il::Range range1;
  il::HMatrixType type;
};

}