QT+=widgets

QMAKE_CXXFLAGS	+= -fopenmp
QMAKE_LFLAGS	+= -fopenmp

CONFIG += c++14


DISTFILES += \
    Scilab/schrod.sce \
    LICENSE \
    README.md\
    Temp/nlse \
    Temp/WIP/domain1dgui.cpp \
    Temp/WIP/equation.cpp\
    Temp/WIP/grid2d.cpp \
    Temp/WIP/integration.cpp \
    Temp/WIP/point2d.cpp \
    Temp/WIP/domain1dgui.h \
    Temp/WIP/equation.h \
    Temp/WIP/grid2d.h \
    Temp/WIP/integration.h \
    Temp/WIP/point2d.h

    

HEADERS += \
    Source/axis.h \
    Source/point.h \
    Source/type.h \
    Source/Domain/domain1d.h \
    Source/Domain/domain2d.h \
    Source/Grid/grid.h \
    Source/Grid/grid1d.h \
    Source/Grid/grid2d.h \
    Source/Grid/gridmanager.h \
    Source/Solver/solver1d.h \
    Source/SolverGui/solver1dgui.h \
    Source/SparseMatrix/sparsematrix.h \
    Source/Domain/domainmanager.h \
    Source/Domain/domain.h

SOURCES += \
    Source/axis.cpp \
    Source/main.cpp \
    Source/point.cpp \
    Source/Domain/domain1d.cpp \
    Source/Domain/domain2d.cpp \
    Source/Grid/grid.cpp \
    Source/Grid/grid1d.cpp \
    Source/Grid/grid2d.cpp \
    Source/Grid/gridmanager.cpp \
    Source/Solver/solver1d.cpp \
    Source/SolverGui/solver1dgui.cpp \
    Source/SparseMatrix/sparsematrix.cpp \
    Source/Domain/domainmanager.cpp \
    Source/Domain/domain.cpp
