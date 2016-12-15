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
    Temp/WIP/point2d.h \
    TODO \
    Temp/TEST_SCHRODINGER1D_main

HEADERS += \
    Source/Axis/axis.h \
    Source/Axis/linearaxis.h \
    Source/Axis/nonlinearaxis.h \
    Source/Function/PDE/functionpdeindexvirtual.h \
    Source/Function/PDE/functionpdevirtual.h \
    Source/Function/functionvirtual.h \
    Source/Grid/Base/gridbase.h \
    Source/Grid/Base/gridmanagerbase.h \
    Source/Grid/grid.h \
    Source/Grid/grid1d.h \
    Source/Grid/grid2d.h \
    Source/Grid/gridmanager.h \
    Source/Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrix.h \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.h \
    Source/Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.h \
    Source/Matrix/Matrix/RowMatrix/rowmatrix.h \
    Source/Matrix/Matrix/RowMatrix/rowmatrixvirtual.h \
    Source/Matrix/Matrix/matrix.h \
    Source/Matrix/MatrixAlgorithm/matrixalgorithm.h \
    Source/Matrix/SparseMatrix/sparsematrix.h \
    Source/Matrix/corematrix.h \
    Source/PDE/GUI/1D/pdegui1d.h \
    Source/PDE/GUI/pdeguivirtual.h \
    Source/PDE/Linear/1D/heat1d.h \
    Source/PDE/Linear/1D/pdelinear1dvirtual.h \
    Source/PDE/Linear/1D/schrodinger1d.h \
    Source/PDE/Linear/pdelinearvirtual.h \
    Source/PDE/pdevirtual.h \
    Source/Solver/Linear/Base/linearsolverbase.h \
    Source/Solver/Linear/linearsolver.h \
    Source/Solver/NonLinear/nonlinearsolver.h \
    Source/Solver/solver.h \
    Source/Systeme/systemevirtual.h \
    Source/debugclass.h \
    Source/frame.h \
    Source/point.h \
    Source/type.h \
    Source/Function/PDE/Schrodinger/NL/1D/schrodingernlint1d.h \
    Source/Function/PDE/Schrodinger/NL/1D/schrodingernlbound1d.h

SOURCES += \
    Source/Axis/axis.cpp \
    Source/Axis/linearaxis.cpp \
    Source/Axis/nonlinearaxis.cpp \
    Source/Function/PDE/functionpdeindexvirtual.cpp \
    Source/Function/PDE/functionpdevirtual.cpp \
    Source/Function/functionvirtual.cpp \
    Source/Grid/Base/gridbase.cpp \
    Source/Grid/Base/gridmanagerbase.cpp \
    Source/Grid/grid.cpp \
    Source/Grid/grid1d.cpp \
    Source/Grid/grid2d.cpp \
    Source/Grid/gridmanager.cpp \
    Source/Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.cpp \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrix.cpp \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.cpp \
    Source/Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.cpp \
    Source/Matrix/Matrix/RowMatrix/rowmatrix.cpp \
    Source/Matrix/Matrix/RowMatrix/rowmatrixvirtual.cpp \
    Source/Matrix/Matrix/matrix.cpp \
    Source/Matrix/MatrixAlgorithm/matrixalgorithm.cpp \
    Source/Matrix/SparseMatrix/sparsematrix.cpp \
    Source/Matrix/corematrix.cpp \
    Source/PDE/GUI/1D/pdegui1d.cpp \
    Source/PDE/GUI/pdeguivirtual.cpp \
    Source/PDE/Linear/1D/heat1d.cpp \
    Source/PDE/Linear/1D/pdelinear1dvirtual.cpp \
    Source/PDE/Linear/1D/schrodinger1d.cpp \
    Source/PDE/Linear/pdelinearvirtual.cpp \
    Source/PDE/pdevirtual.cpp \
    Source/Solver/Linear/Base/linearsolverbase.cpp \
    Source/Solver/Linear/linearsolver.cpp \
    Source/Solver/NonLinear/nonlinearsolver.cpp \
    Source/Solver/solver.cpp \
    Source/Systeme/systemevirtual.cpp \
    Source/debugclass.cpp \
    Source/frame.cpp \
    Source/main.cpp \
    Source/point.cpp \
    Source/type.cpp \
    Source/Function/PDE/Schrodinger/NL/1D/schrodingernlint1d.cpp \
    Source/Function/PDE/Schrodinger/NL/1D/schrodingernlbound1d.cpp


