#/bin/sh
rm src/*.o
g++ -c -O3 -std=gnu++14 -o src/point.o src/point.cpp
g++ -c -O3 -std=gnu++14 -o src/binaryFind.o src/point.cpp
g++ -c -O3 -std=gnu++14 -o src/getSections.o src/point.cpp
g++ -c -O3 -std=gnu++14 -o src/mutualWind2.o src/point.cpp
g++ -c -O3 -std=gnu++14 -o src/localWrithe.o src/point.cpp
g++ -c -O3 -std=gnu++14 -o src/mainFileDI.o src/point.cpp
g++ -O3 -std=gnu++14 -o WritheDI_calc src/point.o src/binaryFind.o src/getSections.o src/mutualWind2.o src/localWrithe.o src/mainFileDI.o
