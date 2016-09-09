QT+=widgets

QMAKE_CXXFLAGS	+= -fopenmp
QMAKE_LFLAGS	+= -fopenmp

CONFIG += c++11


SOURCES += \
    main.cpp \
    grid1d.cpp \
    domain1d.cpp \
    solver1d.cpp \
    solver1dgui.cpp \
    gridmanager.cpp

DISTFILES += \
    nlse \
    schrod.sce

HEADERS += \
    grid1d.h \
    domain1d.h \
    solver1d.h \
    solver1dgui.h \
    gridmanager.h
