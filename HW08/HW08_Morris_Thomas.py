"""
Author: Thomas Morris | tcm1998@rit.edu
Date: 5 April 2020

=======================================

We model the line we seek as:

A * x + B * y + C = 0;
This is the cannonical form, used in linear algebra and many other places.

In our case we are going to use:
A * x + B * y - rho = 0;

We do this because, in a second we are going to constrain A and B in such
a way that rho becomes the distance from the origin to the line.

We need to find A, B, and rho.

However, this is three parameters.  For a line, we know we only need two parameters here.
(For example, slope and intercept when using the form y = mx + b.

So, the form   A * x + B * y - rho = 0;
has 1 too many parameters.

To simplify things, we will assume that
A = cos(theta), and B = sin(theta).

These are called the "directed cosines" of a line.  They give the unit vectors for the line,
based on the angle that the lines goes in.

Then A and B are tied together, and the only two parameters we need are theta, and rho.

So, writing this out again, the model of the line we have is:

cosine(theta) * x + sine(theta) * y - rho = 0;

The angle, theta, it turns out, is the angle from the origin to the closest point
on the line.

Now, IF ONLY WE COULD FIND THE BEST VALUES OF theta AND rho!!

Okay, so anytime someone says, "best" we have do ask the question, "what do we mean by best?"
Or, "Best How?"

In our case, we want the best line to be the one that minimizes the total distance from all of
the points, to the line we want.  That gives the best line -- the one that minimizes the distance
from all of the points to the line.  OR, since there are 6 points, a fixed number of points,
we could minimize the average absolute distance.

We know from lecture 5a&b (or so) that the distance from a point (x,y) to the line
Ax + By + C = 0 is:

dst = abs( Ax + By + C ) / sqrt( A^2 + B^2 )

In the case of each and every point:
   A = cos(theta)
   B = sin(theta)
   C = rho


Procedure:  Guess and Adjust
Step 1.   Fix values of theta and rho.

Step 2.   Find the distance of all the points to the line.

Step 3A.  Try values of rho = rho +/- some change, alpha.
          If the change decreases the total distance of the points to the line,
          then keep making that change.

Step 3B.  Try values of theta = theta +/- some change, alpha.
          If the change decreases the total distance of the points to the line,
          then keep making that change.

Step 4:   Find the current best distance to of the points to the line.
          Compare this to the answer found in Step 2.
          If the difference is <= 0.005  # --> an arbitrary stopping point Dr. Kinsman picked.
              then exit.
          otherwise,
              alpha <-- 0.95 * alpha     # --> alpha gets smaller
              go back to step 2.

          Alternatively, we can stop if the increment we use to change things by gets too small.
          In other words, when alpha gets too small.

As this progresses, it changes the values of the parameters in a direction which decreases the
error -- it goes in the direction that descents the gradients of equal error.

This is an approximation for gradient descent.

Note:  The choice of alpha <-- 0.95 * alpha, uses 0.95.
       This is an arbitrary value.
       The 0.95 determines how fast alpha shrinks.
       This is called THE LEARNING RATE.

In this case, we have two hyper-parameters that are used.
1.  The main one is the learning rate, 0.95.
2.  The other one is the stopping criterion, of 0.005.
"""

import numpy as np
from matplotlib import pyplot as plt
import sys
from math import sqrt, pow, cos, sin

INITIAL_THETA = 15  # TODO Used to be 58
INITIAL_RHO = 9  # TODO Used to be 10.75
INITIAL_ALPHA = 5  # TODO Used to be 4.5
MINIMUM_ALPHA_DELTA = 0.025
MINIMUM_DIST_CHANGE = 0.005  # TODO He originally had this value hard-coded.
LEARNING_RATE = 0.50  # TODO This should be 0.95 according to the MatLab code.


def dst_from_pts_to_line(xys, theta, rho):
    points = xys.shape[1]
    alpha = cos(theta)
    beta = sin(rho)
    total = 0
    for idx in range(points):
        total += abs(xys[0][idx] * alpha + xys[1][idx] * beta - rho) \
               / sqrt(pow(alpha, 2) + pow(beta, 2))

    return total / points


def gradient_descent_fit_through_a_line():
    """Given a set of points, find a line through them, using gradient decent.
    :return:
    """
    # X and Y points.
    xys = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [10, 9, 8, 7, 6, 5, 4, 3, 2]])

    # Initialize the line parameters and learning rate.
    theta = INITIAL_THETA
    rho = INITIAL_RHO
    alpha = INITIAL_ALPHA

    # TODO plot stuff, maybe?
    # draw_line_on_graphs(xys, theta, rho)
    # draw_params_on_graphs(theta, rho)
    # draw_line_on_graphs(xys, theta, rho)
    # draw_params_on_graphs(theta, rho)

    # Save the distance at the start of step 2 for later.
    original_distance = dst_from_pts_to_line(xys, theta, rho)
    print("Starting Distance = {0}\n".format(original_distance))

    # Save the distances for analysis.
    point_distances = [original_distance]

    while True:
        # Step 2. Find the distance of all points to the line.
        current_point_dist = dst_from_pts_to_line(xys, theta, rho)

        # Step 3A. Check a new equation +/- some value to rho.
        # If the distance to points decreases, keep making that change.
        decreased_rho_dist = dst_from_pts_to_line(xys, theta, rho - alpha)
        increased_rho_dist = dst_from_pts_to_line(xys, theta, rho + alpha)

        step_for_rho = alpha
        dst_best_yet = current_point_dist
        if decreased_rho_dist < current_point_dist:
            # Search in the minus direction
            step_for_rho = -alpha
            dst_best_yet = decreased_rho_dist
            rho = rho + step_for_rho  # TAKE THE STEP
        elif increased_rho_dist < current_point_dist:
            # Search in the positive alpha direction.
            step_for_rho = alpha
            dst_best_yet = increased_rho_dist
            rho = rho + step_for_rho  # TAKE THE STEP

        # Look ahead for rho ->
        while True:
            next_point_dist = dst_from_pts_to_line(xys, theta, rho + step_for_rho)

            if next_point_dist < dst_best_yet:
                # This is still improving the distance, so do it again.
                rho = rho + step_for_rho
                dst_best_yet = next_point_dist
                # TODO graph stuff, maybe?
                # draw_line_on_graphs(xys, theta, rho)
                # draw_params_on_graphs(theta, rho)
                # capture_graph(fn_out, 'GIF', 32, 'a', 0.25, 1)
            else:
                # This step would make it worse, so stop.
                break

        # Recalculate the distance of all points to the line.
        current_point_dist = dst_from_pts_to_line(xys, theta, rho)

        # Step 3B. Check a new equation +/- some value to theta.
        # If the distance to points decreases, keep making that change.
        decreased_theta_dist = dst_from_pts_to_line(xys, theta - alpha, rho)
        increased_theta_dist = dst_from_pts_to_line(xys, theta - alpha, rho)

        step_for_theta = alpha
        dst_best_yet = current_point_dist
        if decreased_theta_dist < current_point_dist:
            # Search in the minus direction
            step_for_theta = -alpha
            dst_best_yet = decreased_theta_dist
            theta = theta + step_for_theta  # TAKE THE STEP
        elif increased_theta_dist < current_point_dist:
            # Search in the positive alpha direction.
            step_for_theta = alpha
            dst_best_yet = increased_theta_dist
            theta = theta + step_for_theta  # TAKE THE STEP

        # Look ahead for theta ->
        while True:
            next_point_dist = dst_from_pts_to_line(xys, theta + step_for_theta, rho)

            if next_point_dist < dst_best_yet:
                # This is still improving the distance, so do it again.
                theta = theta + step_for_theta
                dst_best_yet = next_point_dist
                # TODO graph stuff, maybe?
                # draw_line_on_graphs(xys, theta, rho)
                # draw_params_on_graphs(theta, rho)
                # capture_graph(fn_out, 'GIF', 32, 'a', 0.25, 1)
            else:
                # This step would make it worse, so stop.
                break

        # If theta is negative, add 180 to make it positive and negate rho.
        if theta < 0:
            theta = theta + 180
            rho = -rho

        # Step 4. Compare the new line to what we had in step 2.
        #
        # If this is better than step 2 by a significant amount, decrease alpha
        # and return to step 2.
        #
        # Otherwise, exit.
        dst_step_4 = dst_from_pts_to_line(xys, theta, rho)

        # Save the distance for analysis.
        point_distances.append(dst_step_4)

        # TODO graph stuff, maybe?
        # draw_line_on_graphs(xys, theta, rho)
        # draw_params_on_graphs(theta, rho)
        # drawnow
        # capture_graph(fn_out, 'GIF', 32, 'a', 0.25, 1)  # Dr. K

        # Exit if the program is working in reverse.
        if original_distance < dst_step_4:
            sys.exit('Distance is getting worse, not better.')

        # Stop the loop if this iteration made little or no difference.
        if abs(original_distance - dst_step_4) <= MINIMUM_DIST_CHANGE:
            break

        # If we haven't quit yet, decrease alpha, but if alpha drops below a
        # minimum value, quit anyway.
        alpha = LEARNING_RATE * alpha
        if alpha > MINIMUM_ALPHA_DELTA:
            continue
        else:
            break

    print("Rho:\t{0}\nTheta:\t{1}".format(rho, theta))
    print("A:\t{0}\nB:\t{1}".format(cos(theta), sin(theta)))
    print("Final Distance = {0}".format(dst_from_pts_to_line(xys, theta, rho)))
    print("\nIterations = {0}".format(len(point_distances)))

    plt.xlabel('Iteration')
    plt.ylabel('Average Absolute Distance to the Line')
    plt.title('Average Point Distance At Each Iteration')
    plt.plot(range(len(point_distances)), point_distances, 'b')
    plt.show()


def main():
    if LEARNING_RATE >= 1.0:
        sys.exit('Learning Rate cannot be greater then or equal to 1.0')
    elif LEARNING_RATE <= 0:
        sys.exit('Learning Rate cannot be less then or equal to 0.0')

    gradient_descent_fit_through_a_line()


if __name__ == "__main__":
    main()
