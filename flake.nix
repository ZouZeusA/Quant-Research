{
  description = "Quant-Research Python dev environment (Nix + direnv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # or pinned commit
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

      in {
        # Lightweight shell (no compilers → faster)
        devShells.default = pkgs.mkShell {
          name = "quant-research-python-shell";

          packages = [
            pkgs.python314
            pkgs.poetry
            # pythonEnv
            # pkgs.git if not installed system or user wide
            # pkgs.direnv if not installed system or user wide
          ];

          shellHook = ''
            # echo "🐍 Python environment ready — $(python --version)"
            # export PATH="${pkgs.python314}/bin:$PATH"
            # export POETRY_VIRTUALENVS_IN_PROJECT=true
            # export POETRY_VIRTUALENVS_CREATE=true
            # export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

            # # If .venv doesn't exist, create it once automatically
            # if [ ! -d ".venv" ]; then
            #   echo "🔧 Creating poetry environment..."
            #   poetry install --no-root
            # fi

            # # Activate Poetry's virtualenv
            # source .venv/bin/activate
            # echo "✅ Poetry virtualenv activated."
      '';
        };
      });
}
