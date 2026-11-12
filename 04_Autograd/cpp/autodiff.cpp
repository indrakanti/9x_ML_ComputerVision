#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
  double data{0.0};
  double grad{0.0};
  std::vector<ValuePtr> parents;
  std::function<void()> backward{[] {}};
};

ValuePtr make_value(double data) {
  auto value = std::make_shared<Value>();
  value->data = data;
  return value;
}

ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs) {
  auto out = make_value(lhs->data + rhs->data);
  out->parents = {lhs, rhs};
  out->backward = [lhs, rhs, out] {
    lhs->grad += out->grad;
    rhs->grad += out->grad;
  };
  return out;
}

ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs) {
  auto out = make_value(lhs->data * rhs->data);
  out->parents = {lhs, rhs};
  out->backward = [lhs, rhs, out] {
    lhs->grad += rhs->data * out->grad;
    rhs->grad += lhs->data * out->grad;
  };
  return out;
}

ValuePtr power(const ValuePtr& base, double exponent) {
  auto out = make_value(std::pow(base->data, exponent));
  out->parents = {base};
  out->backward = [base, out, exponent] {
    base->grad += exponent * std::pow(base->data, exponent - 1.0) * out->grad;
  };
  return out;
}

void build_topology(const ValuePtr& node, std::set<const Value*>& visited,
                    std::vector<ValuePtr>& topo) {
  if (visited.count(node.get()) != 0U) {
    return;
  }
  visited.insert(node.get());
  for (const auto& parent : node->parents) {
    build_topology(parent, visited, topo);
  }
  topo.push_back(node);
}

void backward(const ValuePtr& root) {
  std::set<const Value*> visited;
  std::vector<ValuePtr> topo;
  build_topology(root, visited, topo);

  root->grad = 1.0;
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward();
  }
}

struct Results {
  double loss;
  double grad_x;
  double grad_y;
};

Results compute_results() {
  const auto x = make_value(2.0);
  const auto y = make_value(3.0);
  const auto xy = multiply(x, y);
  const auto sum = add(xy, x);
  const auto loss = power(sum, 2.0);
  backward(loss);
  return {loss->data, x->grad, y->grad};
}

bool near(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}

int self_test() {
  const Results r = compute_results();
  if (!near(r.loss, 64.0) || !near(r.grad_x, 64.0) || !near(r.grad_y, 32.0)) {
    std::cerr << "autodiff self-test failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void print_json() {
  const Results r = compute_results();
  std::cout << std::setprecision(17)
            << "{"
            << "\"loss\":" << r.loss << ","
            << "\"grad_x\":" << r.grad_x << ","
            << "\"grad_y\":" << r.grad_y
            << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    return self_test();
  }
  if (argc == 2 && std::string(argv[1]) == "--json") {
    print_json();
    return EXIT_SUCCESS;
  }
  std::cout << "Usage: cv9x_autodiff --self-test|--json\n";
  return EXIT_SUCCESS;
}
