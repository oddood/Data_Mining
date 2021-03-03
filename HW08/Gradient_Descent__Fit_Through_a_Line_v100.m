

function Gradient_Descent__Fit_Through_a_Line_v100( )
% Given a set of points, find a line through them, using gradient decent.
%
%  March 21, 2020
%  Thomas B. Kinsman
%
%
    vers = get_fn_version( mfilename() );   % Dr. Kinsman routine.

    xys = [      2     3     4     5     6     7     8     9    10 ;
                10     9     8     7     6     5     4     3     2 ];
     
    
    % We model the line we seek as:
    %
    % A * x + B * y + C = 0;           
    % This is the cannonical form, used in linear algebra and many other places.
    %
    % In our case we are going to use:
    % A * x + B * y - rho = 0;
    %
    % We do this because, in a second we are going to constrain A and B in such 
    % a way that rho becomes the distance from the origin to the line.
    %
    % We need to find A, B, and rho.
    %
    % However, this is three parameters.  For a line, we know we only need two parameters here.
    % (For example, slope and intercept when using the form y = mx + b.
    %
    % So, the form   A * x + B * y - rho = 0;
    % has 1 too many parameters.
    %
    % To simplify things, we will assume that 
    % A = cos(theta), and B = sin(theta).
    % 
    % These are called the "directed cosines" of a line.  They give the unit vectors for the line,
    % based on the angle that the lines goes in.
    %
    % Then A and B are tied together, and the only two parameters we need are theta, and rho.
    %
    % So, writing this out again, the model of the line we have is:
    % 
    % cosine(theta) * x + sine(theta) * y - rho = 0;
    %
    % The angle, theta, it turns out, is the angle from the origin to the closest point
    % on the line.  
    %
    % Now, IF ONLY WE COULD FIND THE BEST VALUES OF theta AND rho!! 
    %
    %
    % Okay, so anytime someone says, "best" we have do ask the question, "what do we mean by best?"
    % Or, "Best How?"
    %
    % In our case, we want the best line to be the one that minimizes the total distance from all of 
    % the points, to the line we want.  That gives the best line -- the one that minimizes the distance 
    % from all of the points to the line.  OR, since there are 6 points, a fixed number of points,
    % we could minimize the average absolute distance.
    %
    % We know from lecture 5a&b (or so) that the distance from a point (x,y) to the line
    % Ax + By + C = 0 is:
    % 
    % dst = abs( Ax + By + C ) / sqrt( A^2 + B^2 )
    %
    % In the case of each and every point:
    %   A = cos(theta)
    %   B = sin(theta)
    %   C = rho
    %
    %
    % Procedure:  Guess and Adjust
    % Step 1.   Fix values of theta and rho.
    %
    % Step 2.   Find the distance of all the points to the line.
    %
    % Step 3A.  Try values of rho = rho +/- some change, alpha.
    %           If the change decreases the total distance of the points to the line,
    %           then keep making that change.
    %
    % Step 3B.  Try values of theta = theta +/- some change, alpha.
    %           If the change decreases the total distance of the points to the line,
    %           then keep making that change.
    %
    % Step 4:   Find the current best distance to of the points to the line.
    %           Compare this to the answer found in Step 2.
    %           If the difference is <= 0.005  # --> an arbitrary stopping point Dr. Kinsman picked.
    %               then exit.
    %           otherwise,
    %               alpha <-- 0.95 * alpha     # --> alpha gets smaller
    %               go back to step 2.
    %           
    %           Alternatively, we can stop if the increment we use to change things by gets too small.
    %           In other words, when alpha gets too small.
    %
    %
    % As this progresses, it changes the values of the parameters in a direction which decreases the 
    % error -- it goes in the direction that descents the gradients of equal error.
    % This is an approximation for gradient descent.
    %
    % Note:  The choice of alpha <-- 0.95 * alpha, uses 0.95.
    %        This is an arbitrary value.
    %        The 0.95 determines how fast alpha shrinks.
    %        This is called THE LEARNING RATE.
    %         
    % In this case, we have two hyper-parameters that are used.
    % 1.  The main one is the learning rate, 0.95.
    % 2.  The other one is the stopping criterion, of 0.005.
    %
    % ===============================================================================
    %
    % Okay, now that I have a plan.  Now let's get coding in Matlab....
    %
    %
    % I copy all of the above comments and duplicate them,
    % and then I start filling in code.
    %
    % It is an iterative technique, and I circle back to the start many times, to 
    % revisit how I implement the algorithm.
    %
    % I will make changes to my initial plan along the way...
    %
INITIAL_THETA           = 58;
INITIAL_RHO             = 10.75;
INITIAL_ALPHA           = 4.5;
MINIMUM_ALPHA_DELTA     = 0.025;
LEARNING_RATE           = 0.90;


    figure( 'Position', [630 350 800 450], 'Color', 'w' );

    if ( LEARNING_RATE >= 1.0 )
        error('Learning Rate cannot be greater then or equal to 1.0');
    elseif ( LEARNING_RATE <= 0 )
        error('Learning Rate cannot be less then or equal to 0.0');
    end
         
    % Step 1.   Set values of theta and rho.
    theta   =   INITIAL_THETA;
    rho     =   INITIAL_RHO;
    alpha   =   INITIAL_ALPHA;
    
    draw_line_on_graphs( xys, theta, rho );
    draw_params_on_graphs( theta, rho );
    draw_line_on_graphs( xys, theta, rho );
    draw_params_on_graphs( theta, rho );
    

    fn_out = sprintf('Fig__Fitting_Line_Problem%s.png', vers );
    capture_graph(fn_out,'PNG', 32, 'w', 1, 1 );                                % Dr. K routine to save the figure.

    fn_out = sprintf('Fig__Fitting_Line_Problem%s.jpg', vers );
    capture_graph(fn_out,'JPG', 32, 'w', 1, 1 );                                % Dr. K routine to save the figure.

    fn_out = sprintf('Fig__Fitting_Line_Using_gradient_descent%s.gif', vers );
    capture_graph(fn_out,'GIF', 32, 'w', 1, 1 );   
    
    while ( 1 )
        %
        % Step 2.   Find the distance of all the points to the line.
        dst_step_2  = dst_from_pts_to_line( xys, theta, rho );

        %
        % Step 3A.  Try values of rho = rho +/- some change, alpha.
        %           If the change decreases the total distance of the points to the line,
        %           then keep making that change.
        dst_step_3a_minus   = dst_from_pts_to_line( xys, theta, rho-alpha );
        dst_step_3a_plus    = dst_from_pts_to_line( xys, theta, rho+alpha );

        if ( dst_step_3a_minus < dst_step_2 )
            % Search in the minus  direction.
            step_for_rho        = -alpha;
            dst_best_yet        =  dst_step_3a_minus;
            rho                 =  rho + step_for_rho;      % TAKE THE STEP
        elseif ( dst_step_3a_plus < dst_step_2 )
            % Search in the positive alpha direction.
            step_for_rho        = +alpha;
            dst_best_yet        =  dst_step_3a_plus;
            rho                 =  rho + step_for_rho;      % TAKE THE STEP
        else
            dst_best_yet        = dst_step_2;
            step_for_rho        = alpha;                    % Has to be something.
                                                            % Do not take a step.
        end

        while( 1 )
            % Look ahead:
            dst_step_3a     = dst_from_pts_to_line( xys, theta, rho+step_for_rho );

            if ( dst_step_3a < dst_best_yet )
                % Still going down, so take the step:
                rho             = rho + step_for_rho;    % Rem: if step_for_rho is negative, this decraments.
                dst_best_yet    = dst_step_3a;
                
                draw_line_on_graphs( xys, theta, rho );
                draw_params_on_graphs( theta, rho );
                capture_graph(fn_out,'GIF', 32, 'a', 0.25, 1 );  
            else 
                break;
            end
        end

        %
        % Step 3B.  Try values of theta = theta +/- some change, alpha.
        %           If the change decreases the total distance of the points to the line,
        %           then keep making that change.
        dst_step_3b         = dst_from_pts_to_line( xys, theta, rho );
        dst_step_3b_minus   = dst_from_pts_to_line( xys, theta-alpha, rho );
        dst_step_3b_plus    = dst_from_pts_to_line( xys, theta+alpha, rho );

        if ( dst_step_3b_minus < dst_step_3b )
            % Search in the minus    alpha direction.
            step_for_theta      = -alpha;
            dst_best_yet        =  dst_step_3b_minus;
            theta               =  theta + step_for_theta;  % TAKE THE PREVIOUS STEP
        elseif ( dst_step_3b_plus < dst_step_3b )
            % Search in the positive alpha direction.
            step_for_theta      = +alpha;
            dst_best_yet        =  dst_step_3b_plus;
            theta               =  theta + step_for_theta;  % TAKE THE PREVIOUS STEP
        else
            dst_best_yet    = dst_step_3b;
            step_for_theta  = alpha;                        % Must be something.
                                                            % Do not take a step.
        end
        
        
        
        while( 1 )
            % LOOK AHEAD:
            dst_step_3b     = dst_from_pts_to_line( xys, theta+step_for_theta, rho );

            if ( dst_step_3b < dst_best_yet )
                theta           = theta + step_for_theta;    % Rem: if step_for_theta is negative, this decraments.
                dst_best_yet    = dst_step_3b;
                % Handle the wrap-around for degrees:
                if ( theta > 180 )
                    theta = theta - 360;
                end
                if ( theta < -180 )
                    theta = theta + 360;
                end
                
                draw_line_on_graphs( xys, theta, rho );
                draw_params_on_graphs( theta, rho );
                capture_graph(fn_out,'GIF', 32, 'a', 0.25, 1 );                                % Dr. K routine to save the figure.
            else
                break;
            end
        end

        % Flip the line from the negative angle a positive angle.
        % This negates rho, whatever that is:
        if ( theta < 0 )
            theta = theta + 180;
            rho   = -rho;
        end

        %
        % Step 4:   Find the current best distance to of the points to the line.
        %
        %           Compare this to the answer found in Step 2.
        %           If the difference is <= 0.005  # --> an arbitrary stopping point Dr. Kinsman picked.
        %               then exit.
        %           otherwise,
        %               alpha <-- 0.95 * alpha     # --> alpha gets smaller
        %               go back to step 2.
        %
        %
        dst_step_4  = dst_from_pts_to_line( xys, theta, rho );

        draw_line_on_graphs( xys, theta, rho );
        draw_params_on_graphs( theta, rho );
        drawnow;
        capture_graph(fn_out,'GIF', 32, 'a', 0.25, 1 );                            % Dr. K routine to save the figure.
   
        fprintf('Rho = %+9.5f  Theta = %+9.6f  ', rho, theta );
        fprintf('Alpha = %+6.5f  Avg Dist = %8.7f\n', alpha, dst_step_4 );
        
        if ( dst_step_2 < dst_step_4 )
            warning('Distance is getting worse, not better.');
        end
        
        alpha = LEARNING_RATE * alpha;
        if ( alpha > MINIMUM_ALPHA_DELTA )
            continue;
        else
            break;
        end
    end

    draw_line_on_graphs( xys, theta, rho );
    draw_params_on_graphs( theta, rho );
    capture_graph(fn_out,'GIF', 32, 'a', 5, 1 );                                % Dr. K routine to save the figure.
    capture_graph(fn_out,'GIF', 32, 'a', 5, 1 );                                % Dr. K routine to save the figure. 

    dst_to_origin = dst_from_pts_to_line( [ 0 ; 0 ], theta, rho );
    
    fprintf('Answer:\n');
    fprintf('A                           = %+7.5f \n',          cosd(theta) );
    fprintf('B                           = %+7.5f \n',          sind(theta) );
    fprintf('theta                       = %+7.5f degrees\n',   theta);
    fprintf('rho                         = %+7.5f \n',          rho );
    fprintf('distance of line to origin  = %+7.5f \n',          dst_to_origin );
    
end



function avg_dist = dst_from_pts_to_line( xys, theta, rho )

    n_pts   =  size( xys, 2 );
    A       =  cosd( theta );
    B       =  sind( theta );
    C       = -rho;

%     ttl_dist = 0;
%     
%     for pt_idx = 1 : n_pts
%         ttl_dist = ttl_dist + abs( A * xys(1,pt_idx) + B * xys(2,pt_idx) + C ) / sqrt( A^2 + B^2 );
%     end
    
    % Vectorized in Matlab, this is all done in the following vector command:
%     ttl_dist = sum( abs( A * xys(1,:) + B * xys(2,:) + C )) / sqrt( A^2 + B^2 );
    
    % But the sin^2 + cos^ 2 = 1, so don't bother dividing by that:
    
    % Vectorized in Matlab, this is all done in the following vector command:
    ttl_dist = sum( abs( A * xys(1,:) + B * xys(2,:) + C ));
    
    avg_dist = ttl_dist / n_pts;
end


function draw_line_on_graphs( xys, theta, rho )
persistent top_axis;

    if ( isempty( top_axis ) )
        top_axis = axes( 'Position', [0.075    0.0875    0.4    0.85] );
    end
    axes( top_axis );
    set( gca, 'Position', [0.075    0.0875    0.4    0.85] );
    
    hold off;
    % Add the points:
    plot(xys(1,:), xys(2,:), 'bo', 'MarkerFaceColor', 'y', 'MarkerSize', 14 );

    axis( [ 0 16 0 16 ] );
    grid on;
    set(gca,'GridAlpha', 0.3 );
    axis equal;

    xlabel('X',                 'FontSize', 22 );
    ylabel('Y',                 'FontSize', 22 );
    title( 'Cartesian Space',   'FontSize', 22 );

    hold on;
    
    A       =  cosd( theta );
    B       =  sind( theta );
    C       = -rho;
    
    if ( abs(theta) > 1 )
        % Rem: Ax + By + C = 0
        % By = -(Ax + C)
        % y  = -(Ax + C)/B

        xs =  [-20 : 20];
        ys = -(A*xs + C)/B;
    else
        % Rem: Ax + By + C = 0
        % Ax = -(By + C) 
        % x  = -(By + C)/A
        ys =  [0 : 20];
        xs = -(B * ys + C)/A;        
    end
    
    hold on;
    % Add the line:
    plot( xs, ys, 'k-' );
    
    % Find the one point on the line that is closest to the origin.
    % The line from the origin to this line is perpendicular, so (theta + 90)
    % applies.

    D = -B/A;                       % Perpendicular to Ax + By + C
    E = 1;
    F = 0;                          % line through origin has no offset.
    
    % Solve for the new point:
    % [ A B C ] * [ X, Y, 1]' = 0;
    % [ D E F ] * [ X, Y, 1]' = 0;
    %
    % Solve for X and Y ... 
    %
    % [ A B ] * [ X, Y ]' = [-C ];
    % [ D E ] * [ X, Y ]' = [ 0 ];  % Because F = 0
    %
    M   = [  A B ; 
             D E ];
    RHS = [ -C ; F ];
    
    XY  = M \ RHS;
    
    plot( [0 XY(1)], [0 XY(2)], 'rs-', 'LineWidth', 1.5 );
    
    % Put a value of rho on the line:
    if ( rho > 1.5 )
        txt = sprintf('\\rho=%4.3f', rho );
        text( [XY(1)*2/3], [XY(2)*2/3], txt, 'FontSize', 18, 'Color', 'r' );
    end
    
    txt = sprintf('\\theta=%5.1f', theta );
    text( 1, 1/2, txt, 'FontSize', 18, 'Color', 'b' );
    
    
    axis( [ 0 16 0 16 ] );
    
end


function draw_params_on_graphs( theta, rho )
persistent last_theta;
persistent last_rho;
persistent last_color_number;
persistent bot_axis;

    if ( isempty( bot_axis ) )
        bot_axis = axes('Position', [0.575    0.0875    0.4    0.8] );
    end

    axes( bot_axis );
    set( gca, 'Position', [0.575    0.0875    0.4    0.8]  );
    
colors  = 'yyy';

    grid on;
    set(gca, 'GridAlpha', 0.5 );
    set(gca, 'GridColor', 'c' );
    
    if ( isempty( last_theta ) )
        plot( theta, rho, 'ko' );
        last_color_number = 1;
    else
        color = colors( last_color_number );
        plot( [ last_theta ], [ last_rho ], 'ko-', 'MarkerFaceColor', 'k' );
        hold on;
        plot( [ last_theta, theta ], [ last_rho, rho ],  'k-', 'MarkerFaceColor', color );
        plot( [ theta ], [ rho ], 'ko', 'MarkerFaceColor', color );
    end
    xlabel('Theta = \Theta (degrees)', 'FontSize', 22 );
    ylabel('Rho = \rho',               'FontSize', 22 );
    title( 'Parameter Space',          'FontSize', 22 );

    last_theta               = theta;
    last_rho                 = rho;
    last_color_number        = last_color_number + 1;
    if ( last_color_number > length( colors ) )
        last_color_number = 1;
    end
    hold on;
    
    
    axis( [ 35, 55,  6,  10 ] );
end
