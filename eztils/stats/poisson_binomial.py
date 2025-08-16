import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_pmf(bernoullis):
    """
    Plot the Poisson-Binomial PMF of overall accuracy given heterogeneous Bernoulli success probs.
    
    Args:
        bernoullis: list or array of success probabilities for each trial
    
    Produces:
        A stem plot of the accuracy PMF with mean and ±1 std marked.
    """
    n = len(bernoullis)
    
    # Poisson-Binomial PMF via FFT
    def poisson_binomial_pmf(probs):
        n = len(probs)
        N = n + 1
        omega = np.exp(-2j * np.pi * np.arange(N) / N)
        G = np.ones(N, dtype=complex)
        for p in probs:
            G *= (1 - p) + p * omega
        pmf = np.fft.ifft(G).real
        pmf = np.maximum(pmf, 0)
        pmf /= pmf.sum()
        return pmf
    
    pmf_k = poisson_binomial_pmf(bernoullis)
    k = np.arange(n + 1)
    acc_vals = k / n
    
    # Mean and std of accuracy
    mean_acc = np.sum(acc_vals * pmf_k)
    std_acc = np.sqrt(np.sum(((acc_vals - mean_acc) ** 2) * pmf_k))
    
    # Plot
    plt.stem(acc_vals, pmf_k, basefmt=" ", label="PMF")
    plt.axvline(mean_acc, color="red", linestyle="--", label=f"Mean = {mean_acc:.3f}")
    plt.axvline(mean_acc - std_acc, color="green", linestyle=":", 
                label=f"Mean ± Std = {mean_acc-std_acc:.3f}, {mean_acc+std_acc:.3f}")
    plt.axvline(mean_acc + std_acc, color="green", linestyle=":")
    plt.xlabel("Accuracy (k/n)")
    plt.ylabel("Probability")
    plt.title("PMF of Overall Accuracy (Poisson-Binomial)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    
    return mean_acc, std_acc

if __name__ == '__main__':
  # Example usage
  bernoullis = [0.8, 0.9, 0.1, 0.1, 0.2, 0.3, 0.43, 0.19, 0.99, 0.8, 0.8, 0.1]
  plot_accuracy_pmf(bernoullis)
