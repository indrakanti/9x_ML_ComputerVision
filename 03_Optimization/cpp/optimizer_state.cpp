#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct MomentumState {
  double value;
  double velocity;
};

struct AdamState {
  double value;
  double first_moment;
  double second_moment;
  int step;
};

double gradient_descent_step(double value, double gradient, double learning_rate) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("learning_rate must be > 0");
  }
  return value - learning_rate * gradient;
}

MomentumState momentum_step(MomentumState state, double gradient,
                            double learning_rate, double beta = 0.9) {
  if (learning_rate <= 0.0 || beta < 0.0 || beta >= 1.0) {
    throw std::invalid_argument("invalid momentum hyperparameters");
  }
  state.velocity = beta * state.velocity + gradient;
  state.value -= learning_rate * state.velocity;
  return state;
}

AdamState adam_step(AdamState state, double gradient, double learning_rate,
                    double beta1 = 0.9, double beta2 = 0.999,
                    double epsilon = 1e-8) {
  if (learning_rate <= 0.0 || beta1 < 0.0 || beta1 >= 1.0 ||
      beta2 < 0.0 || beta2 >= 1.0 || epsilon <= 0.0) {
    throw std::invalid_argument("invalid Adam hyperparameters");
  }

  ++state.step;
  state.first_moment =
      beta1 * state.first_moment + (1.0 - beta1) * gradient;
  state.second_moment =
      beta2 * state.second_moment + (1.0 - beta2) * gradient * gradient;

  const double first_hat =
      state.first_moment / (1.0 - std::pow(beta1, state.step));
  const double second_hat =
      state.second_moment / (1.0 - std::pow(beta2, state.step));

  state.value -=
      learning_rate * first_hat / (std::sqrt(second_hat) + epsilon);
  return state;
}

double quadratic_gradient(double value) { return 2.0 * (value - 3.0); }

struct Results {
  double gd_one_step;
  double momentum_two_steps;
  double adam_200_steps;
};

Results compute_results() {
  const double gd = gradient_descent_step(10.0, quadratic_gradient(10.0), 0.1);

  MomentumState momentum{10.0, 0.0};
  for (int i = 0; i < 2; ++i) {
    momentum = momentum_step(momentum, quadratic_gradient(momentum.value), 0.01);
  }

  AdamState adam{10.0, 0.0, 0.0, 0};
  for (int i = 0; i < 200; ++i) {
    adam = adam_step(adam, quadratic_gradient(adam.value), 0.1);
  }

  return {gd, momentum.value, adam.value};
}

bool near(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}

int self_test() {
  const Results r = compute_results();
  if (!near(r.gd_one_step, 8.6) || !near(r.momentum_two_steps, 9.5968) ||
      std::abs(r.adam_200_steps - 3.0) >= 0.01) {
    std::cerr << "optimizer self-test failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void print_json() {
  const Results r = compute_results();
  std::cout << std::setprecision(17)
            << "{"
            << "\"gd_one_step\":" << r.gd_one_step << ","
            << "\"momentum_two_steps\":" << r.momentum_two_steps << ","
            << "\"adam_200_steps\":" << r.adam_200_steps
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
  std::cout << "Usage: cv9x_optimizer_state --self-test|--json\n";
  return EXIT_SUCCESS;
}
