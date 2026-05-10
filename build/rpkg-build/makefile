# -----------------------------------------------------------------------------
# G2S R interface Makefile (sub-makefile under build/R-build)
# -----------------------------------------------------------------------------

R				?= R
CURL			?= curl -L
GIT				?= git
PYTHON3			?= python3

PKG_NAME		:= G2S
PKG_DIR			:= g2s
VERSION_FILE	:= ../../version
VERSION			:= $(shell test -f $(VERSION_FILE) && cat $(VERSION_FILE) || echo 0.1.0)

CPPZMQ_HPP_URL	:= https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp
LIBZMQ_H_URL	:= https://raw.githubusercontent.com/zeromq/libzmq/master/include/zmq.h
JSONCPP_REPO	:= https://github.com/open-source-parsers/jsoncpp.git

SRC_CPP			:= ../../src_interfaces/R_interface.cpp
HDR_PUBLIC_1	:= ../../include_interfaces/R_interface.hpp
HDR_PUBLIC_DIRS	:= ../../include ../../include_interfaces
R_API_FILE		:= g2s.R

PKG_R_DIR		:= $(PKG_DIR)/R
PKG_SRC_DIR		:= $(PKG_DIR)/src
PKG_INC_DIR		:= $(PKG_DIR)/inst/include
PKG_MAN_DIR		:= $(PKG_DIR)/man

all: build

# -----------------------------------------------------------------------------
# Directory and scaffolding
# -----------------------------------------------------------------------------
dirs:
	@mkdir -p $(PKG_R_DIR) $(PKG_SRC_DIR) $(PKG_INC_DIR) $(PKG_MAN_DIR)

$(PKG_DIR)/DESCRIPTION: | dirs
	@echo "Package: $(PKG_NAME)" > $@
	@echo "Type: Package" >> $@
	@echo "Title: Interface to the G2S Geostatistical Server" >> $@
	@echo "Version: $(VERSION)" >> $@
	@echo "Author: Mathieu Gravey [aut, cre]" >> $@
	@echo "Maintainer: Mathieu Gravey <gravey.mathieu@gmail.com>" >> $@
	@echo "Description: Provides an R interface to the G2S geostatistical computation framework using Rcpp and ZeroMQ." >> $@
	@echo "License: GPL-3" >> $@
	@echo "Encoding: UTF-8" >> $@
	@echo "Depends: R (>= 3.5.0)" >> $@
	@echo "Imports: Rcpp" >> $@
	@echo "LinkingTo: Rcpp" >> $@
	@echo "SystemRequirements: ZeroMQ, jsoncpp" >> $@
	@echo "RoxygenNote: 7.3.1" >> $@
	@echo "✅ Created DESCRIPTION"

$(PKG_DIR)/NAMESPACE: | dirs
	@echo "useDynLib($(PKG_NAME), .registration=TRUE)" > $@
	@echo "importFrom(Rcpp, evalCpp)" >> $@
	@echo "export(g2s)" >> $@
	@echo "✅ Created NAMESPACE"

$(PKG_SRC_DIR)/Makevars: | dirs
	@printf "PKG_CPPFLAGS = -Wno-deprecated-declarations -DJSON_NO_SYSTEM_INCLUDE\n" > $@
	@printf "PKG_CXXFLAGS = -std=c++17 -I../inst/include/g2s_json -I../inst/include -I../../include -I../../include_interfaces\n" >> $@
	@printf "PKG_LIBS = -lzmq\n" >> $@
	@printf "PKG_LDFLAGS = -Wl,-force_load,$(PKG_SRC_DIR)/jsoncpp.o\n" >> $@
	@echo "✅ Wrote src/Makevars"

$(PKG_INC_DIR)/zmq.hpp: | dirs
	@$(CURL) "$(CPPZMQ_HPP_URL)" -o "$(PKG_INC_DIR)/zmq.hpp"
	@$(CURL) "$(LIBZMQ_H_URL)" -o "$(PKG_INC_DIR)/zmq.h"
	@echo "✅ Downloaded cppzmq headers"

rcpp_exports:
	@echo "🔧 Generating Rcpp attribute bindings..."
	Rscript -e "Rcpp::compileAttributes('$(PKG_DIR)')"
	@test -f $(PKG_DIR)/src/RcppExports.cpp && echo "✅ RcppExports.cpp generated" || echo "⚠️ No RcppExports.cpp created (check for // [[Rcpp::export]])"


copy_headers: | dirs
	@cp $(HDR_PUBLIC_1) $(PKG_INC_DIR)/
	@for d in $(HDR_PUBLIC_DIRS); do \
		test -d $$d && cp -R $$d/* "$(PKG_INC_DIR)/" || true; \
	done
	@echo "✅ Copied project headers"

jsoncpp:
	@echo "📦 Fetching and preparing jsoncpp..."
	rm -rf jsoncpp
	git clone --depth 1 https://github.com/open-source-parsers/jsoncpp.git
	cd jsoncpp && python3 amalgamate.py
	@mkdir -p $(PKG_INC_DIR)/g2s_json
	@cp -r jsoncpp/dist/json/* $(PKG_INC_DIR)/g2s_json/
	@cp jsoncpp/dist/jsoncpp.cpp $(PKG_SRC_DIR)/
	#@sed -i '' 's|"json_tool.h"|"g2s_json/json_tool.h"|' $(PKG_SRC_DIR)/jsoncpp.cpp || true
	@sed -i '' 's|json/|g2s_json/|' $(PKG_SRC_DIR)/jsoncpp.cpp || true
	@for f in $(PKG_INC_DIR)/*; do \
		sed -i '' 's|json/|g2s_json/|g' "$$f"; \
	done
	@echo "✅ Added local jsoncpp amalgamation (namespaced as g2s_json)"

# -----------------------------------------------------------------------------
# Main build sequence
# -----------------------------------------------------------------------------
prep: $(PKG_DIR)/DESCRIPTION $(PKG_DIR)/NAMESPACE dirs copy_headers $(PKG_INC_DIR)/zmq.hpp jsoncpp $(PKG_SRC_DIR)/Makevars
	@cp $(SRC_CPP) $(PKG_SRC_DIR)/
	@cp $(R_API_FILE) $(PKG_R_DIR)/
	@echo "✅ Prepared R package structure under $(PKG_DIR)"

build: prep rcpp_exports
	@$(R) CMD build $(PKG_DIR)
	@$(R) CMD INSTALL --build $(PKG_DIR)
	@echo "✅ Built source and binary package"

source: prep rcpp_exports
	@$(R) CMD build $(PKG_DIR)
	@echo "✅ Source tar.gz ready"

binary: prep rcpp_exports
	@$(R) CMD INSTALL --build $(PKG_DIR)
	@echo "✅ Binary .tgz built"

clean:
	rm -rf $(PKG_DIR)/src $(PKG_DIR)/inst/include jsoncpp $(PKG_DIR)/g2s $(PKG_DIR)/R
	rm -f $(PKG_NAME)_*.tar.gz $(PKG_NAME)_*.tgz $(PKG_DIR)/DESCRIPTION $(PKG_DIR)/NAMESPACE
	@echo "🧹 Cleaned R build"

.PHONY: all dirs prep build source binary clean copy_headers jsoncpp rcpp_exports
