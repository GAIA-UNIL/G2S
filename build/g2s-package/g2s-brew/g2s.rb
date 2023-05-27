class G2s < Formula
  desc "Toolbox for geostatistical simulations"
  homepage "https://gaia-unil.github.io/G2S/"
  version "x.y.z"
  url "https://github.com/GAIA-UNIL/G2S/archive/COMMIT_HASH.tar.gz"
  sha256 "COMMIT_HASH256"
  license "GPL-3.0-only"
    
  option "with-intel", "Use intel compiler if available (x86_64 only)"
  option "with-intel-static", "Use intel compiler if available (x86_64 only) and compile files without intel dependencies"
  
  # Add dependencies
  depends_on "cppzmq"
  depends_on "fftw" if !build.with? 'intel' or !build.with? 'intel-static'
  depends_on "jsoncpp"
  depends_on "libomp" if !build.with? 'intel' or !build.with? 'intel-static'
  depends_on "zeromq"
  uses_from_macos "curl"
  uses_from_macos "zlib"

  def source_intel
    output = `sh -c '. /opt/intel/oneapi/setvars.sh>/dev/null; env'`
    output.each_line do |line|
      key, value = line.strip.split("=", 2)
      if value.nil?
        ENV[key] = ""
      else
        ENV[key] = value
      end
    end
  end

  def install

    intel_path=nil

    if build.with? "intel" or build.with? "intel-static"
      source_intel
      intel_path = which("icpx")

      if intel_path.nil?
        intel_path = which("icpc")
      end
    end

    gccLibPath=`gcc -print-libgcc-file-name | xargs dirname`
    puts Formula["zlib"].opt_lib

    extraFlagForStatic="";
    if  build.with? "intel-static"
      extraFlagForStatic="AS_STATIC=1"
    end
  
    if !intel_path.nil? && Hardware::CPU.arch == :x86_64
      cd "build" do
          # Run "make c++ -j"
          system "make", "intel", "-j", extraFlagForStatic, "CXXFLAGS=-fopenmp -DWITH_MKL -I#{Formula["jsoncpp"].opt_include} -I#{Formula["cppzmq"].opt_include} -std=c++17",
          "LIB_PATH= -fuse-ld=lld -L#{Formula["zlib"].opt_lib} -lz -L#{Formula["cppzmq"].opt_lib} -L#{Formula["jsoncpp"].opt_lib}"

          system "make", "c++-server", "-j"
        end

        # Copy g2s_server and other from the c++-build folder to the brew bin folder
        bin.install "build/g2s-package/g2s-brew/g2s"
        libexec.install "build/c++-build/g2s_server"
        libexec.install "build/intel-build/echo"
        libexec.install "build/intel-build/qs"
        libexec.install "build/intel-build/nds"
        libexec.install "build/intel-build/ds-l"
        libexec.install "build/intel-build/errorTest"
        libexec.install "build/intel-build/auto_qs"
        libexec.install "build/algosName.config"
        if File.exist?("build/intel-build/g2s_cuda.so")
          libexec.install "build/intel-build/g2s_cuda.so"
        end

    else

        if ENV["CC"] =~ /clang/
          cxxflags = "-Xclang -fopenmp"
        else
          cxxflags = "-fopenmp"
        end

        cd "build" do
          # Run "make c++ -j"
          system "make", "c++", "-j", "CXXFLAGS=#{cxxflags} -I#{Formula["fftw"].opt_include}", \
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
        if File.exist?("build/c++-build/g2s_cuda.so")
          libexec.install "build/c++-build/g2s_cuda.so"
        end

    end
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
