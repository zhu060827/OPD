@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo 哈工大大创结题版工程系统启动脚本
echo ==========================================
echo 当前目录：%cd%
echo.
echo 第 1 步：检查并安装依赖...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo 依赖安装失败。请确认 VSCode 终端已经进入正确 Python 虚拟环境。
  pause
  exit /b 1
)
echo.
echo 第 2 步：启动 Flask 后端...
echo 浏览器打开：http://127.0.0.1:5000
echo 如果要停止服务，请在这个窗口按 Ctrl+C。
echo.
python app.py
pause
