@echo off
set PHPRC=%~dp0
echo Running Migrations with MySQL & SQLite support...
php artisan migrate
php artisan db:seed --class=FaceProfileSeeder
pause
