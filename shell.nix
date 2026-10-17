{ pkgs ? import <nixpkgs> {} }:

let
  wheelLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    fontconfig
    freetype
    expat
    libdrm
    libglvnd
    mesa
    libGLU
    xorg.libX11
    xorg.libXrender
    xorg.libXext
    xorg.libXcursor
    xorg.libXi
    xorg.libXfixes
    xorg.libXinerama
    xorg.libXxf86vm
    xorg.libSM
    xorg.libICE
    xorg.libXt
  ];
  libPath = pkgs.lib.makeLibraryPath wheelLibs;
in
pkgs.mkShell {
  packages = with pkgs; [
    python313
    uv
    ruff
    xvfb-run
  ];

  env = {
    UV_PYTHON_PREFERENCE = "only-system";
    UV_PYTHON = "${pkgs.python313}/bin/python3.13";
  };

  shellHook = ''
    export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
  '';
}
