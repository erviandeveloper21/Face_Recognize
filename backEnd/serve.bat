@echo off
set PHPRC=%~dp0
echo Starting Laravel Development Server with MySQL & SQLite support...
php artisan serve
pause
