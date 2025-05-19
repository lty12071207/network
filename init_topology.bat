@echo off
:: 设置代码页为65001，支持UTF-8编码
chcp 65001 >nul

echo 请输入是否执行清理（yes/no，默认yes）：
set /p choice=
:: 只有输入no时才跳过清理
if /i "%choice%"=="no" goto skip

:: 执行清理任务

:: 1、创建api目录并写入route.json文件
if not exist api mkdir api
echo [ ] > api\route.json
echo 文件已创建在api\route.json

:: 2、切换到utils目录
cd utils

:: 3、执行Python脚本
python config_general.py
python topo_handle.py

echo 清理任务已执行完成
cd ..

:skip
:: 执行主任务
python app.py

echo Finished
pause