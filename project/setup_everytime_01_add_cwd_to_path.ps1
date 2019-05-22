
# add current directory to PATH (so libraries can be found)

$Env:PATH=$ENV:PATH + $PSScriptRoot
$Env:PATH=$ENV:PATH + $PSScriptRoot + '/lib'

