class G2s < Formula
  desc "Toolbox for geostatistical simulations."
  homepage "https://gaia-unil.github.io/G2S/"
  version "x.y.z"
  url "https://github.com/GAIA-UNIL/g2s/archive/COMMIT_HASH.tar.gz"
  sha256 "COMMIT_HASH256"
  license "GPL-3.0-only"

  # Add dependencies
  depends_on "cppzmq"
  depends_on "curl"
  depends_on "fftw"
  depends_on "jsoncpp"
  depends_on "libomp"
  depends_on "zeromq"
  depends_on "zlib"

  def install
    cd "build" do
      # Run "make c++ -j"
      system "make", "c++", "-j", "CXXFLAGS=-Xclang -fopenmp -I#{Formula["fftw"].opt_include}", \
          "LIB_PATH=-L#{Formula["fftw"].opt_lib} -L#{Formula["libomp"].opt_lib} -lomp"
    end

    # Copy g2s_server and other from the c++-build folder to the brew bin folder
    bin.install "build/g2s-package/g2s-brew/g2s"
    (prefix/"g2s_bin").install "build/c++-build/g2s_server"
    (prefix/"g2s_bin").install "build/c++-build/echo"
    (prefix/"g2s_bin").install "build/c++-build/qs"
    (prefix/"g2s_bin").install "build/c++-build/nds"
    (prefix/"g2s_bin").install "build/c++-build/ds-l"
    (prefix/"g2s_bin").install "build/c++-build/errorTest"
    (prefix/"g2s_bin").install "build/c++-build/auto_qs"
    (prefix/"g2s_bin").install "build/algosName.config"

    # bash_completion.install "build/g2s-package/g2s-brew/g2s-completion.sh"
    # zsh_completion.install "build/g2s-package/g2s-brew/g2s-completion.zsh"
    # fish_completion.install "build/g2s-package/g2s-brew/g2s-completion.fish"
  end

  service do
    run [opt_bin/"g2s", "server", "-kod"]
    keep_alive true
  end

  test do
    begin
      pid = fork do
        exec bin/"g2s", "server"
      end
      sleep 3
    ensure
      Process.kill("TERM", pid)
      Process.wait(pid)
    end
  end
end

