from graphviz import Digraph

def create_digital_twin_flowchart():
    # Create a new directed graph
    dot = Digraph(comment='Digital Twin System Flowchart')
    dot.attr(rankdir='TB')  # Top to Bottom direction
    
    # Set default node style
    dot.attr('node', shape='box', style='filled')
    
    # Main components
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Initialization')
        with c.subgraph(name='cluster_init') as i:
            i.attr(label='__init__()')
            i.attr('node', fillcolor='lightgreen')
            i.node('init', 'Initialize\nDigital Twin')
            i.node('pygame', 'pygame.init()')
            i.node('serial', 'serial.Serial()\n(115200 baud)')
            i.node('physics', 'Initialize Physics\nParameters\n(theta=0, theta_dot=0\nL=0.5m, g=9.81m/s²)')
            i.node('motor', 'Initialize Motor\nState\n(acc=0, vel=0, pos=0)')
            i.node('actions', 'Initialize Action\nConfiguration\n(duration: 150-200ms\ndirection: left/right)')
            
            # Connect initialization nodes
            i.edge('init', 'pygame')
            i.edge('init', 'serial')
            i.edge('init', 'physics')
            i.edge('init', 'motor')
            i.edge('init', 'actions')
    
    # Lab1_simulation components
    with dot.subgraph(name='cluster_lab1') as c:
        c.attr(label='Lab1_simulation')
        with c.subgraph(name='cluster_lab1_main') as l:
            l.attr(label='Main Loop')
            l.attr('node', fillcolor='lightblue')
            l.node('lab1_init', 'Initialize\nLab1_simulation')
            l.node('lab1_loop', 'Main Loop\n(delta_t = 0.01s)')
            l.node('lab1_step', 'digital_twin.step()')
            l.node('lab1_render', 'digital_twin.render()')
            l.node('lab1_events', 'pygame.event.get()')
            l.node('lab1_key_press', 'Handle Key Press\n(A,S,D,F,J,K,L,;)')
            l.node('lab1_quit', 'pygame.quit()')
            
            # Connect Lab1 nodes
            l.edge('lab1_init', 'lab1_loop')
            l.edge('lab1_loop', 'lab1_step')
            l.edge('lab1_step', 'lab1_render')
            l.edge('lab1_render', 'lab1_events')
            l.edge('lab1_events', 'lab1_key_press')
            l.edge('lab1_events', 'lab1_quit')
            l.edge('lab1_key_press', 'lab1_loop')
    
    # Digital Twin Step Function
    with dot.subgraph(name='cluster_step') as c:
        c.attr(label='Digital Twin Step Function')
        with c.subgraph(name='cluster_step_inner') as s:
            s.attr(label='step()')
            s.attr('node', fillcolor='lightyellow')
            s.node('check_prediction', 'check_prediction_lists()\n(prediction horizon=100)')
            s.node('update_motor', 'Update Motor State\ncurrentmotor_acceleration\ncurrentmotor_velocity\nx_pivot')
            s.node('calc_theta', 'get_theta_double_dot()\nθ̈ = g/L * sin(θ)\n+ R_pulley/L * ẍ')
            s.node('update_theta', 'θ += θ̇ * delta_t')
            s.node('update_theta_dot', 'θ̇ += θ̈ * delta_t')
            s.node('update_x_pivot', 'x_pivot += R_pulley * motor_positions')
            
            # Connect step components
            s.edge('check_prediction', 'update_motor')
            s.edge('update_motor', 'calc_theta')
            s.edge('calc_theta', 'update_theta')
            s.edge('update_theta', 'update_theta_dot')
            s.edge('update_motor', 'update_x_pivot')
    
    # Action Handling
    with dot.subgraph(name='cluster_action') as c:
        c.attr(label='Action Handling')
        with c.subgraph(name='cluster_action_inner') as a:
            a.attr(label='Action Processing')
            a.attr('node', fillcolor='lightpink')
            a.node('check_actions', 'Check actions dictionary\n(valid key?)')
            a.node('perform_action', 'perform_action()\n(direction, duration)')
            
            # Create nested structure for update_motor_accelerations_real
            with a.subgraph(name='cluster_motor_acc') as m:
                m.attr(label='update_motor_accelerations_real()')
                m.node('calc_acc', 'Calculate Motor\nAccelerations\n(acc = ±0.5 m/s²)')
                m.node('update_acc_list', 'Update\nfuture_motor_accelerations\n(prediction horizon=100)')
                m.node('update_vel_list', 'Update\nfuture_motor_velocities\n(v = v₀ + at)')
                m.node('update_pos_list', 'Update\nfuture_motor_positions\n(x = x₀ + vt)')
                
                # Connect nested nodes
                m.edge('calc_acc', 'update_acc_list')
                m.edge('update_acc_list', 'update_vel_list')
                m.edge('update_vel_list', 'update_pos_list')
            
            # Connect action nodes
            a.edge('check_actions', 'perform_action')
            a.edge('perform_action', 'calc_acc')
    
    # Display components
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='Display Components')
        with c.subgraph(name='cluster_render') as r:
            r.attr(label='render()')
            r.attr('node', fillcolor='lightgray')
            r.node('draw_grid', 'draw_grid()\n(background)')
            r.node('draw_pendulum', 'draw_pendulum()\n(theta, L=0.5m)')
            r.node('draw_track', 'Draw Track\n(x_pivot)')
            r.node('draw_indicators', 'draw_indicators()\n(θ̇, θ̈)')
            r.node('draw_info', 'draw_info_panel()\n(θ, θ̇, x_pivot)')
            r.node('draw_keys', 'draw_key_actions()\n(A,S,D,F,J,K,L,;)')
            
            # Connect display nodes
            r.edge('draw_grid', 'draw_pendulum')
            r.edge('draw_pendulum', 'draw_track')
            r.edge('draw_track', 'draw_indicators')
            r.edge('draw_indicators', 'draw_info')
            r.edge('draw_info', 'draw_keys')
    
    # Connect clusters with data flow labels
    dot.edge('init', 'lab1_init', 'Initialize')
    dot.edge('lab1_step', 'check_prediction', 'Step\n(delta_t=0.01s)')
    dot.edge('lab1_render', 'draw_grid', 'Render\n(state)')
    dot.edge('lab1_key_press', 'check_actions', 'Key Press\n(event)')
    dot.edge('update_pos_list', 'check_prediction', 'Update Lists\n(prediction)')
    
    # Save the flowchart
    dot.render('digital_twin_flowchart', format='png', cleanup=True)

if __name__ == '__main__':
    create_digital_twin_flowchart() 