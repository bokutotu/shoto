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

        project = pkgs.haskell-nix.project' {
          src = ./.;
          compiler-nix-name = "ghc912";
        };

      in {
        devShells.default = project.shellFor {
          tools = {
            haskell-language-server = {};
            cabal                  = {};
            hlint                  = {};
            stylish-haskell        = {};
            cabal-gild             = {};
            fast-tags              = {};
            fourmolu               = {};
            hspec-discover         = {};
          };
          buildInputs = [ pkgs.git ];
        };
      });
}
