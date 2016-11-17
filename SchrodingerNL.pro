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
    Source/point.h \
    Source/type.h \
    Source/Domain/domain1d.h \
    Source/Domain/domain2d.h \
    Source/Grid/grid.h \
    Source/Grid/grid1d.h \
    Source/Grid/grid2d.h \
    Source/Grid/gridmanager.h \
    Source/Domain/domainmanager.h \
    Source/Domain/domain.h \
    Source/Solver/Linear/linearsolver.h \
    Source/Matrix/SparseMatrix/sparsematrix.h \
    Source/Matrix/Matrix/matrix.h \
    Source/Matrix/corematrix.h \
    Source/Solver/solver.h \
    Source/Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrix.h \
    Source/Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.h \
    Source/Matrix/Matrix/RowMatrix/rowmatrix.h \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.h \
    Source/Matrix/Matrix/RowMatrix/rowmatrixvirtual.h \
    Source/Matrix/MatrixAlgorithm/matrixalgorithm.h \
    Source/PDE/Linear/pdelinearvirtual.h \
    Source/PDE/Linear/1D/pdelinear1dvirtual.h \
    Source/PDE/pdevirtual.h \
    Source/PDE/Linear/1D/heat1d.h \
    Source/frame.h \
    Source/BoundaryCondition/boundarycondition.h \
    Source/Axis/axis.h \
    Source/Axis/linearaxis.h \
    Source/Axis/nonlinearaxis.h \
    Source/PDE/Linear/1D/schrodinger1d.h \
    Source/PDE/GUI/1D/pdegui1d.h \
    Source/PDE/GUI/pdeguivirtual.h \
    Source/Domain/Base/domainbase.h \
    Source/Domain/Base/domainmanagerbase.h \
    Source/Grid/Base/gridbase.h \
    Source/debugclass.h \
    Source/Solver/NonLinear/nonlinearsolver.h \
    Source/Systeme/systemevirtual.h \
    Source/Function/functionvirtual.h \
    Source/Function/functionpdevirtual.h

SOURCES += \
    Source/main.cpp \
    Source/point.cpp \
    Source/Domain/domain1d.cpp \
    Source/Domain/domain2d.cpp \
    Source/Grid/grid.cpp \
    Source/Grid/grid1d.cpp \
    Source/Grid/grid2d.cpp \
    Source/Grid/gridmanager.cpp \
    Source/Domain/domainmanager.cpp \
    Source/Domain/domain.cpp \
    Source/Matrix/SparseMatrix/sparsematrix.cpp \
    Source/Solver/Linear/linearsolver.cpp \
    Source/Matrix/Matrix/matrix.cpp \
    Source/Matrix/corematrix.cpp \
    Source/type.cpp \
    Source/Solver/solver.cpp \
    Source/Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.cpp \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrix.cpp \
    Source/Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.cpp \
    Source/Matrix/Matrix/RowMatrix/rowmatrix.cpp \
    Source/Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.cpp \
    Source/Matrix/Matrix/RowMatrix/rowmatrixvirtual.cpp \
    Source/Matrix/MatrixAlgorithm/matrixalgorithm.cpp \
    Source/PDE/Linear/pdelinearvirtual.cpp \
    Source/PDE/Linear/1D/pdelinear1dvirtual.cpp \
    Source/PDE/pdevirtual.cpp \
    Source/PDE/Linear/1D/heat1d.cpp \
    Source/frame.cpp \
    Source/BoundaryCondition/boundarycondition.cpp \
    Source/Axis/axis.cpp \
    Source/Axis/linearaxis.cpp \
    Source/Axis/nonlinearaxis.cpp \
    Source/PDE/Linear/1D/schrodinger1d.cpp \
    Source/PDE/GUI/1D/pdegui1d.cpp \
    Source/PDE/GUI/pdeguivirtual.cpp \
    Source/Domain/Base/domainbase.cpp \
    Source/Domain/Base/domainmanagerbase.cpp \
    Source/Grid/Base/gridbase.cpp \
    Source/debugclass.cpp \
    Source/Solver/NonLinear/nonlinearsolver.cpp \
    Source/Systeme/systemevirtual.cpp \
    Source/Function/functionvirtual.cpp \
    Source/Function/functionpdevirtual.cpp
