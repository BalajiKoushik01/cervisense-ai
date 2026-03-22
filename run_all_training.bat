@echo off
echo ===========================================
echo Starting CerviSense-AI Training Pipeline
echo ===========================================
set PYTHONPATH=.
echo.

echo [Phase 1] Self-Supervised Learning Pre-Training (MoCo-v3)
.\cervisense_env\Scripts\python.exe training\train_ssl.py --config configs\ssl_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo Phase 1 Failed! Exiting.
    exit /b %ERRORLEVEL%
)

echo.
echo [Phase 2] Supervised Fine-Tuning
.\cervisense_env\Scripts\python.exe training\train_finetune.py --config configs\finetune_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo Phase 2 Failed! Exiting.
    exit /b %ERRORLEVEL%
)

echo.
echo [Phase 3] Cross-Modal Fusion
.\cervisense_env\Scripts\python.exe training\train_fusion.py --config configs\fusion_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo Phase 3 Failed! Exiting.
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================
echo Training Pipeline Finished Successfully!
echo ===========================================
