{
  description = "Python 3.12 development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            python312Packages.pip
            uv
            sqlite
          ];

          shellHook = ''
            echo "Python 3.12 development environment loaded"
            echo "uv package manager is available"
            python --version
          '';
        };
      }
    );
}
