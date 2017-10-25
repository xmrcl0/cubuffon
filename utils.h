/** @file     utils.h
 *  @brief    Function prototypes for utils.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     09/12/2017
 *  @update   09/21/2017
 */

#include <regex.h>
#include <stdio.h>


/** @brief Verify if a string is a valid natural number.
 *
 *  @param[in] n  Number to be validated
 *  @return Validation status 
 *
 *  @retval  0 Success
 *  @retval  1 Failure
 *  @retval -1 Internal error
 */
int is_natural_num(char *n);


/** @brief Verify if a string is a valid positive number.
 *
 *  @param[in] number Number to be validated
 *  @return Validation status 
 *
 *  @retval  0 Success
 *  @retval  1 Failure
 *  @retval -1 Internal error
 */
int is_positive_num(char *n);
