QT+=widgets

QMAKE_CXXFLAGS	+= -fopenmp
QMAKE_LFLAGS	+= -fopenmp

CONFIG += c++11


DISTFILES += \
    Scilab/schrod.sce \
    LICENSE \
    README.md\
    Temp/nlse\
    WIP/domain1dgui.h \
    WIP/equation.h \
    WIP/grid2d.h \
    WIP/point2d.h \
    WIP/integration.h \
    WIP/domain1dgui.cpp \
    WIP/equation.cpp \
    WIP/integration.cpp \
    WIP/grid2d.cpp \
    WIP/point2d.cpp

HEADERS += \
    include/domain1d.h \
    include/grid1d.h \
    include/gridmanager.h \
    include/solver1d.h \
    include/solver1dgui.h \
    include/axis.h \
    include/grid2d.h \
    include/domain2d.h \
    include/type.h \
    include/point.h \
    include/sparsematrix.h \
    include/grid.h


SOURCES += \
    src/domain1d.cpp \
    src/grid1d.cpp \
    src/gridmanager.cpp \
    src/main.cpp \
    src/grid2d.cpp \
    src/domain2d.cpp \
    src/axis.cpp \
    src/solver1d.cpp \
    src/solver1dgui.cpp \
    src/point.cpp \
    src/sparsematrix.cpp \
    src/grid.cpp
