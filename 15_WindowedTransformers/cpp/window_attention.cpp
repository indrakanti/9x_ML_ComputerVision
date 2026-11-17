#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Artifact {
  int grid_h{0}, grid_w{0}, embed{0}, window{0};
  std::vector<double> wq, wk, wv;
};

std::vector<double> parse(const std::string& text) {
  std::vector<double> out;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) out.push_back(std::stod(token));
  return out;
}

Artifact load(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("unable to open artifact");
  std::unordered_map<std::string,std::string> v;
  std::string line;
  while (std::getline(input,line)) {
    if (line.empty() || line.front()=='#') continue;
    auto p=line.find('=');
    if (p==std::string::npos) throw std::runtime_error("invalid artifact");
    v[line.substr(0,p)]=line.substr(p+1);
  }
  if (std::stoi(v.at("version"))!=1) throw std::runtime_error("bad version");
  Artifact a;
  a.grid_h=std::stoi(v.at("grid_h")); a.grid_w=std::stoi(v.at("grid_w"));
  a.embed=std::stoi(v.at("embed")); a.window=std::stoi(v.at("window"));
  a.wq=parse(v.at("wq")); a.wk=parse(v.at("wk")); a.wv=parse(v.at("wv"));
  return a;
}

std::vector<double> example() {
  std::vector<double> v(32);
  for (std::size_t i=0;i<v.size();++i) v[i]=static_cast<double>(i)/10.0;
  return v;
}

std::vector<double> project(const std::vector<double>& x,
                            const std::vector<double>& w,
                            int dim) {
  std::vector<double> out(static_cast<std::size_t>(dim),0.0);
  for(int o=0;o<dim;++o)
    for(int i=0;i<dim;++i)
      out[static_cast<std::size_t>(o)] +=
        w[static_cast<std::size_t>(o*dim+i)] * x[static_cast<std::size_t>(i)];
  return out;
}

std::vector<double> forward(const Artifact& a,const std::vector<double>& tokens) {
  std::vector<double> output;
  const double scale=std::sqrt(static_cast<double>(a.embed));
  for(int top=0;top<a.grid_h;top+=a.window){
    for(int left=0;left<a.grid_w;left+=a.window){
      std::vector<std::vector<double>> window;
      for(int y=0;y<a.window;++y) for(int x=0;x<a.window;++x){
        int t=(top+y)*a.grid_w+left+x;
        std::vector<double> row;
        for(int d=0;d<a.embed;++d)
          row.push_back(tokens[static_cast<std::size_t>(t*a.embed+d)]);
        window.push_back(row);
      }
      std::vector<std::vector<double>> q,k,v;
      for(const auto& row:window){
        q.push_back(project(row,a.wq,a.embed));
        k.push_back(project(row,a.wk,a.embed));
        v.push_back(project(row,a.wv,a.embed));
      }
      for(const auto& query:q){
        std::vector<double> scores;
        for(const auto& key:k){
          double s=0.0;
          for(int d=0;d<a.embed;++d)
            s+=query[static_cast<std::size_t>(d)]*key[static_cast<std::size_t>(d)];
          scores.push_back(s/scale);
        }
        double m=scores.front();
        for(double s:scores) if(s>m) m=s;
        double total=0.0;
        for(double& s:scores){s=std::exp(s-m);total+=s;}
        for(double& s:scores) s/=total;
        for(int d=0;d<a.embed;++d){
          double z=0.0;
          for(std::size_t i=0;i<v.size();++i)
            z+=scores[i]*v[i][static_cast<std::size_t>(d)];
          output.push_back(z);
        }
      }
    }
  }
  return output;
}

void print_json(const std::string& path){
  auto out=forward(load(path),example());
  std::cout<<std::setprecision(17)<<"{\"values\":[";
  for(std::size_t i=0;i<out.size();++i){if(i)std::cout<<",";std::cout<<out[i];}
  std::cout<<"]}\n";
}
}  // namespace

int main(int argc,char** argv){
  try{
    if(argc==3&&std::string(argv[1])=="--json"){print_json(argv[2]);return 0;}
    if(argc==3&&std::string(argv[1])=="--self-test"){
      return forward(load(argv[2]),example()).size()==32U?0:1;
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
