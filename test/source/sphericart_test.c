#include "sphericart/sphericart.h"

#include <string.h>

int main(int argc, const char* argv[])
{
  (void)argc;
  (void)argv;

  return strcmp(exported_function(), "sphericart") == 0 ? 0 : 1;
}
