@echo off
setlocal
if not defined GW40_MSVC_VCVARS set "GW40_MSVC_VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%GW40_MSVC_VCVARS%" >nul
if errorlevel 1 exit /b %ERRORLEVEL%
if defined GW40_CUDA_HOME set "CUDA_HOME=%GW40_CUDA_HOME%"
if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH%"
if not defined CUDA_HOME set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=12.0"
set "MAX_JOBS=4"
set "DISTUTILS_USE_SDK=1"
set "CL=/bigobj /Zc:preprocessor"
set "TORCH_EXTENSIONS_DIR=%GW40_NATIVE_JIT%"
"%GW40_NATIVE_PYTHON%" -X utf8 -u "%~dp0stages.py" "%~1" "%~2" "%~3"
exit /b %ERRORLEVEL%
