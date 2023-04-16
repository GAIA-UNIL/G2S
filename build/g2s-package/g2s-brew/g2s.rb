class G2s < Formula
  desc "Toolbox for geostatistical simulations"
  homepage "https://gaia-unil.github.io/G2S/"
  url "https://github.com/GAIA-UNIL/G2S/archive/COMMIT_HASH.tar.gz"
  sha256 "COMMIT_HASH256"
  license "GPL-3.0-only"
  version "x.y.z"

  # Add dependencies
  depends_on "cppzmq"
  depends_on "fftw"
  depends_on "jsoncpp"
  depends_on "libomp"
  depends_on "zeromq"
  uses_from_macos "curl"
  uses_from_macos "zlib"

  def install
    cd "build" do
      # Run "make c++ -j"
      system "make", "c++", "-j", "CXXFLAGS=-Xclang -fopenmp -I#{Formula["fftw"].opt_include}", \
          "LIB_PATH=-L#{Formula["fftw"].opt_lib} -L#{Formula["libomp"].opt_lib} -lomp"
    end

    # Copy g2s_server and other from the c++-build folder to the brew bin folder
    bin.install "build/g2s-package/g2s-brew/g2s"
    libexec.install "build/c++-build/g2s_server"
    libexec.install "build/c++-build/echo"
    libexec.install "build/c++-build/qs"
    libexec.install "build/c++-build/nds"
    libexec.install "build/c++-build/ds-l"
    libexec.install "build/c++-build/errorTest"
    libexec.install "build/c++-build/auto_qs"
    libexec.install "build/algosName.config"

    # bash_completion.install "build/g2s-package/g2s-brew/g2s-completion.sh"
    # zsh_completion.install "build/g2s-package/g2s-brew/g2s-completion.zsh"
    # fish_completion.install "build/g2s-package/g2s-brew/g2s-completion.fish"
  end

  service do
    run [opt_bin/"g2s", "server", "-kod"]
    keep_alive true
  end

  test do
    pid = fork do
      exec bin/"g2s", "server"
    end
    sleep 3
  ensure
    Process.kill("TERM", pid)
    Process.wait(pid)
  end
end
