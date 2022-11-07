#include "localWrithe.h"
#include <time.h>
#include <string.h>
#include <string>
#include <iostream>



int main( int argc, const char* argv[] )
{
       if(argc >=2){
	bool check=false;
	std::ifstream myfile;
 	myfile.open(argv[1]);
	std::string output;
	std::vector<point> points;
	if (myfile.is_open()) { 
	  while(std::getline(myfile,output)){
		points.push_back(point(output));
  		}
	}else{
	std::cout<<"Curve data file failed to open";
	}
	if(points.size()>3){
	    check =true;
	}
	myfile.close();
	if(check){
	  localWrithe lw;
	  double wr = lw.DI(points);
	  std::cout <<wr<<"\n";
	}else{
	  std::cout<<"need more than 3 points in the curve\n";
	}
       }else{
	  std::cout<<"must supply a file containing the curve data";
       }
       return 0;
}