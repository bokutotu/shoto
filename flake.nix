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
  };

  outputs = { self, nixpkgs, flake-utils, haskellNix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ haskellNix.overlay ];
          inherit (haskellNix) config;
        };
        
        # GCC 13を使用
        gccForCuda = pkgs.gcc13;

        # hpackでcabalファイルを生成したソースを作成
        src = pkgs.runCommand "shoto-src" {
          nativeBuildInputs = [ pkgs.haskellPackages.hpack ];
        } ''
          cp -r ${./.} $out
          chmod -R +w $out
          cd $out
          hpack shoto/
          hpack isl/
        '';

        project = pkgs.haskell-nix.project' {
          inherit src;
          compiler-nix-name = "ghc912";
          modules = [{
            # システムのcudartライブラリを使用
            packages.shoto.components.library.libs = pkgs.lib.mkForce [];
            packages.shoto.components.tests.shoto-test.libs = pkgs.lib.mkForce [];
          }];
        };
      in {
        devShells.default = project.shellFor {
          tools = {
            haskell-language-server = {};
            cabal                  = {};
            hlint                  = {};
            stylish-haskell        = {};
            fast-tags              = {};
            fourmolu               = {};
            hspec-discover         = {};
            hpack                  = {};
          };
          buildInputs = [
            pkgs.git
            gccForCuda
          ];
          shellHook = ''
            # GCC 13を優先
            export PATH="${gccForCuda}/bin:$PATH"
            export CC="${gccForCuda}/bin/gcc"
            export CXX="${gccForCuda}/bin/g++"
            
            # システムのCUDAを使う
            if [ -n "$CUDA_PATH" ]; then
              export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
              export LIBRARY_PATH="$CUDA_PATH/lib64:$LIBRARY_PATH"
              export C_INCLUDE_PATH="$CUDA_PATH/include:$C_INCLUDE_PATH"
            elif [ -d "/usr/local/cuda" ]; then
              export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
              export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
              export C_INCLUDE_PATH="/usr/local/cuda/include:$C_INCLUDE_PATH"
            fi
            
            # WSL環境用の追加パス
            if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
              export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH"
            fi
          '';
        };
      });
}
