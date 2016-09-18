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
    WIP/domain1dgui.cpp \
    WIP/equation.cpp \
    WIP/grid2d.cpp \
    WIP/point2d.cpp

HEADERS += \
    include/domain1d.h \
    include/grid1d.h \
    include/gridmanager.h \
    include/integration.h \
    include/solver1d.h \
    include/solver1dgui.h \


SOURCES += \
    src/domain1d.cpp \
    src/grid1d.cpp \
    src/gridmanager.cpp \
    src/integration.cpp \
    src/main.cpp \
    src/solver1d.cpp \
    src/solver1dgui.cpp \

