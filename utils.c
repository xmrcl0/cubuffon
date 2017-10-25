/** @file     utils.c
 *  @brief    Utils functions.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     09/12/2017
 *  @update   09/21/2017
 */

#include <utils.h>


int
is_natural_num (char *number)
{
  int r;
  regex_t regex;
  const char *pattern = "^[0-9]+$";

  r = regcomp (&regex, pattern, REG_EXTENDED);
  if (r)
  {
    fprintf (stderr, "Could not compile regex\n");
    return -1;
  }

  r = regexec (&regex, number, 0, NULL, 0);
  if (!r)
  {
    return 1;
  }
  return 0;
}


int
is_positive_num (char *number)
{
  int r;
  regex_t regex;
  const char *pattern = "^[0-9]+\\.?([0-9]+)?$";

  r = regcomp (&regex, pattern, REG_EXTENDED);
  if (r)
  {
    fprintf (stderr, "Could not compile regex\n");
    return -1;
  }

  r = regexec (&regex, number, 0, NULL, 0);
  if (!r)
  {
    return 1;
  }
  return 0;
}
