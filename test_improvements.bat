@echo off
echo Testing DocRAG Performance Improvements
echo =======================================
echo.

echo 1. First test with original settings:
echo ------------------------------------
python main.py --query "Was kannst du mir über Betriebssysteme erzählen?" --model deepseek-v3
echo.
echo.

echo 2. Second test with improved settings (DeepSeek-R1):
echo -------------------------------------------------
python main.py --query "Was kannst du mir über Betriebssysteme erzählen?" --model deepseek-r1 --no_graph
echo.
echo.

echo 3. Third test with o3-mini model:
echo ------------------------------
python main.py --query "Was kannst du mir über Betriebssysteme erzählen?" --model o3-mini --no_graph
echo.
echo.

echo Test complete. Check the timing logs to see the performance improvement.
echo The PERFORMANCE_IMPROVEMENTS.md file has more details on the changes made.