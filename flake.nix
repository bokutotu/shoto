{
  description = "Shoto - A polyhedral compiler";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://cache.iog.io"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ="
    ];
    experimental-features = [ "nix-command" "flakes" ];
  };

  inputs = {
    haskellNix.url  = "github:input-output-hk/haskell.nix";
    nixpkgs.follows = "haskellNix/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    isl = {
      url = "git+https://repo.or.cz/isl.git?ref=refs/tags/isl-0.27";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, haskellNix, isl }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        islOverlay = final: prev:
          let
            islGit = prev.stdenv.mkDerivation {
              pname = "isl";
              version = "0.27";
              src = isl;
              nativeBuildInputs = [
                prev.autoconf
                prev.automake
                prev.libtool
                prev.gnum4
                prev.pkg-config
              ];
              buildInputs = [
                prev.gmp
              ];
              preConfigure = ''
                ./autogen.sh
              '';
              configureFlags = [
                "--with-gmp-prefix=${prev.gmp}"
              ];
              enableParallelBuilding = true;
            };
          in {
            islGit = islGit;
            haskell-nix = prev.haskell-nix // {
              extraPkgconfigMappings = prev.haskell-nix.extraPkgconfigMappings or {} // {
                "isl" = [ "islGit" ];
              };
            };
          };

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ haskellNix.overlay islOverlay ];
          inherit (haskellNix) config;
        };

        isLinux = pkgs.stdenv.isLinux;
        srcRoot = builtins.path {
          path = ./.;
          name = "shoto-root";
          filter = path: type:
            let
              baseName = builtins.baseNameOf path;
            in
              ! (
                baseName == ".git"
                || baseName == ".cabal"
                || baseName == ".ccls-cache"
                || baseName == "dist-newstyle"
                || baseName == "tags"
              );
        };
        

        # hpackでcabalファイルを生成したソースを作成
        mkProjectSrc =
          pkgs.runCommand "shoto-src" {
          nativeBuildInputs = [ pkgs.haskellPackages.hpack ];
        } ''
          cp -r ${srcRoot} $out
          chmod -R +w $out
          cd $out
          hpack shoto/
          hpack isl/
        '';

        src = mkProjectSrc;

        project = pkgs.haskell-nix.project' {
          inherit src;
          compiler-nix-name = "ghc912";
          modules = [{
            # システムのcudartライブラリを使用
            packages.shoto.components.library.libs = pkgs.lib.mkForce [];
            packages.shoto.components.tests.shoto-test.libs = pkgs.lib.mkForce [];
          }];
        };

        shellTools = {
          haskell-language-server = {};
          cabal                  = {};
          hlint                  = {};
          stylish-haskell        = {};
          fast-tags              = {};
          fourmolu               = {};
          hspec-discover         = {};
          hpack                  = {};
          apply-refact           = {};
        };

        baseBuildInputs = [
          pkgs.git
          pkgs.lefthook
          pkgs.pkg-config
          pkgs.islGit
        ];

        mkShell = { projectForShell, withCuda ? true }:
          projectForShell.shellFor {
            tools = shellTools;
            buildInputs = baseBuildInputs;
            shellHook = ''
              # lefthookをインストール
              lefthook install

              export SHOTO_CABAL_PROJECT_FILE="cabal.project"

              libstdcxx_dir="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
              export LD_LIBRARY_PATH="$libstdcxx_dir:$LD_LIBRARY_PATH"
              export LIBRARY_PATH="$libstdcxx_dir:$LIBRARY_PATH"

              # CUDA-enabled shells use the system CUDA installation.
              if [ -n "$CUDA_PATH" ]; then
                export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
                export LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
                export C_INCLUDE_PATH="$CUDA_PATH/include:$CUDA_PATH/targets/x86_64-linux/include:$C_INCLUDE_PATH"
              else
                for cuda_path in /usr/local/cuda /usr/local/cuda-*; do
                  if [ -d "$cuda_path" ]; then
                    export CUDA_PATH="$cuda_path"
                    export LD_LIBRARY_PATH="$cuda_path/lib64:$cuda_path/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
                    export LIBRARY_PATH="$cuda_path/lib64:$cuda_path/targets/x86_64-linux/lib:$LIBRARY_PATH"
                    export C_INCLUDE_PATH="$cuda_path/include:$cuda_path/targets/x86_64-linux/include:$C_INCLUDE_PATH"
                    break
                  fi
                done
              fi
              
              # WSL環境用の追加パス
              if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
                export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH"
              fi

              # Prefer isl 0.27 from nix in this shell
              export PKG_CONFIG_PATH="${pkgs.islGit}/lib/pkgconfig:$PKG_CONFIG_PATH"
            '';
          };
      in {
        devShells = {
          default = mkShell { projectForShell = project; };
        };

        packages = {
          default = project.hsPkgs.shoto.components.library;
        };
      });
}
