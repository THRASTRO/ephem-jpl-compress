@echo off

for %%p in (x y z) do (
	call wt.exe --window "%1" new-tab --profile "Command Prompt" --title "wave-fit %1 p%%p" -d "%CD%" ^
		x64\Release\wave-fit.exe satellites\%1\states\small.p%%p.f64 --gpu-device 1
	sleep 1
	call wt.exe --window "%1" new-tab --profile "Command Prompt" --title "wave-fit %1 v%%p" -d "%CD%" ^
		x64\Release\wave-fit.exe satellites\%1\states\small.v%%p.f64 --gpu-device 0
	sleep 15
)

REM wmic process where name="wave-fit5.exe" CALL setpriority "Below Normal"