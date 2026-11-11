@echo off
set PHPRC=%~dp0
echo Menghapus semua profil wajah dan riwayat scan...
php artisan face:clear
pause
