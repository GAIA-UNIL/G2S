class G2s < Formula
  desc "Toolbox for geostatistical simulations."
  homepage "https://gaia-unil.github.io/G2S/"
  url "https://github.com/GAIA-UNIL/g2s/archive/COMMIT_HASH.tar.gz"
  sha256 "COMMIT_HASH256"
  license "GPL-3.0-only"
  version "x.y.z"

  # Add dependencies
  depends_on "zlib"
  depends_on "fftw"
  depends_on "zmq"
  depends_on "jsoncpp"
  depends_on "cppzmq"
  depends_on "curl"

  def install

    cd "build" do
      # Run "make c++ -j"
      system "make", "c++", "-j", "CXXFLAGS=-I#{Formula["fftw"].opt_include}", "LIB_PATH=-L#{Formula["fftw"].opt_lib}"
    end

    # Copy g2s_server and other from the c++-build folder to the brew bin folder
    (bin).install "build/g2s-package/g2s-brew/g2s" 
    (bin/"g2s_bin").install "build/c++-build/g2s_server" 
    (bin/"g2s_bin").install "build/c++-build/echo" 
    (bin/"g2s_bin").install "build/c++-build/qs" 
    (bin/"g2s_bin").install "build/c++-build/nds" 
    (bin/"g2s_bin").install "build/c++-build/ds-l" 
    (bin/"g2s_bin").install "build/c++-build/errorTest" 
    (bin/"g2s_bin").install "build/c++-build/auto_qs"

    # bash_completion.install "build/g2s-package/g2s-brew/g2s-completion.sh"
    # zsh_completion.install "build/g2s-package/g2s-brew/g2s-completion.zsh"
    # fish_completion.install "build/g2s-package/g2s-brew/g2s-completion.fish"
  end

  def plist
    <<~EOS
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
        <key>Label</key>
        <string>#{plist_name}</string>
        <key>ProgramArguments</key>
        <array>
          <string>#{bin}/g2s</string>
          <string>server</string>
          <string>-kod</string>
          <string>${FLAGS}</string>
        </array>
        <key>RunAtLoad</key>
        <false/>
        <key>KeepAlive</key>
        <true/>
        <key>StandardErrorPath</key>
        <string>/tmp/#{plist_name}.err</string>
        <key>StandardOutPath</key>
        <string>/tmp/#{plist_name}.out</string>
      </dict>
      </plist>
    EOS
  end


  test do
    # Test if the software is functioning correctly
    system "#{bin}/g2s", "--help"
  end
end
