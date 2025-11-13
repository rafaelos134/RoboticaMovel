-------------------------------------------------------------------
--                                                               --
-- LASER ANGLES FROM UNPROJECTED HOKUYO'S FAST SENSOR            --
-- WITH DUAL CAMERAS                                             --
--                                                               --
-------------------------------------------------------------------
-- Adapted from fastHokuyo native sensor script                  --
-- Alysson Ribeiro da Silva - Federal University of Minas Gerais --
-------------------------------------------------------------------
--                                                               --
-- This script does an unprojection of the laser beans           --
-- collision in the tangent of the camera view.                  --
-- These angles are sent through the API along their             --
-- distances. If nothing is detected along the path, then        --
-- the max detection distance is put for that laser bean         --
-- to be returned.                                               --
--                                                               --
-------------------------------------------------------------------
--                                                               --
-- String signals sent:                                          --
--          hokuyo_range_data - contains all the range data      --
--          hokuyo_angle_data - contains all angles from the     --
--                              unprojected points from the      --
--                              camera's view                    --
--                                                               --
-------------------------------------------------------------------


-- Unproject sensor beans
-- input: total laser beans
-- output: all laser beans angles in radians
function calc_tangent_unprojection_angles(laser_beans_count)
    -- get to total amount of laser beans from one camera
    len_range_2 = math.floor(laser_beans_count / 2.0)
    
    -- rotate 90 degrees right
    start_angle = -math.pi / 2.0
    angles = {}
    
    for i=0,laser_beans_count-1,1 do
        -- unproject the left camera laser beans from tangent
        if i < len_range_2 then
            index = i
            
            -- normalize index for projection
            lrp = index / (len_range_2 - 1.0)
            
            -- project tangent into curve and get its angle
            tan_map = lrp * 2.0 - 1.0
            tan_ang = math.atan(tan_map)
            
            -- rotate angle to match the robot's frame
            tan_ang = tan_ang + math.pi / 4.0 + start_angle
            
            -- add angle into return array
            table.insert(angles, tan_ang)
            
        -- unproject the right camera laser beans
        else
            index = i - len_range_2
            
            -- normalize index for projection
            lrp = index / (len_range_2 - 1.0)
            
            -- project tangent into curve and get its angle
            tan_map = lrp * 2.0 - 1.0
            tan_ang = math.atan(tan_map)
            
            -- rotate angle to match the robot's frame
            tan_ang = tan_ang + math.pi * 3.0/4.0 + start_angle
            
            -- add angle into return array
            table.insert(angles, tan_ang)
        end
    end
    
    return angles
end

-- Initialize all variables for this script to work properly
-- input: None
-- output: None
function sysCall_init()
    local self=sim.getObjectHandle(sim.handle_self)
    
    -- handlers for this sensor
    visionSensor1Handle=sim.getObjectHandle("fastHokuyo_sensor1")
    visionSensor2Handle=sim.getObjectHandle("fastHokuyo_sensor2")
    joint1Handle=sim.getObjectHandle("fastHokuyo_joint1")
    joint2Handle=sim.getObjectHandle("fastHokuyo_joint2")
    sensorRef=sim.getObjectHandle("fastHokuyo_ref")
    local collection=sim.createCollection(0)
    sim.addItemToCollection(collection,sim.handle_all,-1,0)
    sim.addItemToCollection(collection,sim.handle_tree,self,1)
    
    -- set cameras configuration
    -- it includes:
    --     camera span
    --     camera rotation
    --     sensor max distance
    --     scanning angle for each camera
    maxScanDistance=5
    maxScanDistance_=maxScanDistance*0.9999
    scanningAngle=180*math.pi/180
    
    sim.setObjectInt32Param(visionSensor1Handle,sim.visionintparam_entity_to_render,collection)
    sim.setObjectInt32Param(visionSensor2Handle,sim.visionintparam_entity_to_render,collection)
    sim.setObjectFloatParam(visionSensor1Handle,sim.visionfloatparam_far_clipping,maxScanDistance)
    sim.setObjectFloatParam(visionSensor2Handle,sim.visionfloatparam_far_clipping,maxScanDistance)
    sim.setObjectFloatParam(visionSensor1Handle,sim.visionfloatparam_perspective_angle,scanningAngle/2)
    sim.setObjectFloatParam(visionSensor2Handle,sim.visionfloatparam_perspective_angle,scanningAngle/2)

    -- set each camera rotation to 45 degrees so both will cover
    -- all the necessary area for a view span of 90 degrees
    sim.setJointPosition(joint1Handle,-scanningAngle/4)
    sim.setJointPosition(joint2Handle,scanningAngle/4)
    
    -- drawing helpers
    red={1,0,0}
    lines=sim.addDrawingObject(sim.drawing_lines,1,0,-1,10000,nil,nil,nil,red)
    
    -------------------------------------------------------------------
    -- SET THIS FLAG TO TRUE IF YOU WANT TO DRAW THE LASER DATA      --
    -- IT IS NOT RECOMMENDED TO DO SO BECAUSE IT WILL SLOW           --
    -- YOUR SIMULATOR'S MAIN LOOP SPEED IN APPROXIMATELY 5ms         --
    -------------------------------------------------------------------
    showLines=true
    -------------------------------------------------------------------
    
    -- distances array used to return every laser range data
    dists={}
    
    -- calculate all angles by unprojecting the camera's view
    -- this should be done only once for eficiency
    -- the total beans variable can be obtained by adding
    -- both cameras width, which are 342 + 342
    total_beans = 684
    angles = calc_tangent_unprojection_angles(total_beans)
end

-- Cleanup all system variables 
-- input: None
-- output: None
function sysCall_cleanup() 
    sim.removeDrawingObject(lines)
end 

-- Update function for this script, it calculate all colision points to draw
-- the laser correctly. Furthermore, it will also mount the distances array
-- to be sent through the API 
-- input: None
-- output: None
function sysCall_sensing() 
    dists={}
    
    if notFirstHere then
        -- We skip the very first reading
        sim.addDrawingObjectItem(lines,nil)
        
        -- get data from cameras which is an array
        -- of width = 342 for each one
        r1,t1,u1=sim.readVisionSensor(visionSensor1Handle)
        r2,t2,u2=sim.readVisionSensor(visionSensor2Handle)
        
        -- both m1 and m2 are matrices to draw
        -- the laser beans relative to the world
        -- inside the simulator frame
        m1=sim.getObjectMatrix(visionSensor1Handle,-1)
        m2=sim.getObjectMatrix(visionSensor2Handle,-1)

        if u1 then
            p={0,0,0}
            p=sim.multiplyVector(m1,p)
            t={p[1],p[2],p[3],0,0,0}
            for j=0,u1[2]-1,1 do
                for i=0,u1[1]-1,1 do
                    w=2+4*(j*u1[1]+i)
                    v1=u1[w+1]
                    v2=u1[w+2]
                    v3=u1[w+3]
                    v4=u1[w+4]
                    
                    if (v4<maxScanDistance_) then
                        -- v4 holds the laser distance
                        -- to the detected obstacle
                        table.insert(dists, v4)
                    else
                        table.insert(dists, maxScanDistance_)
                    end
                    
                    if showLines then
                        p={v1,v2,v3}
                        p=sim.multiplyVector(m1,p)
                        t[4]=p[1]
                        t[5]=p[2]
                        t[6]=p[3]
                        sim.addDrawingObjectItem(lines,t)
                    end
                end
            end
        end
        if u2 then
            p={0,0,0}
            p=sim.multiplyVector(m2,p)
            t={p[1],p[2],p[3],0,0,0}
            for j=0,u2[2]-1,1 do
                for i=0,u2[1]-1,1 do
                    w=2+4*(j*u2[1]+i)
                    v1=u2[w+1]
                    v2=u2[w+2]
                    v3=u2[w+3]
                    v4=u2[w+4]
                    
                    if (v4<maxScanDistance_) then
                        -- v4 holds the laser distance
                        -- to the detected obstacle
                        table.insert(dists, v4)
                    else    
                        table.insert(dists, maxScanDistance_)
                    end
                    
                    if showLines then
                        p={v1,v2,v3}
                        p=sim.multiplyVector(m2,p)
                        t[4]=p[1]
                        t[5]=p[2]
                        t[6]=p[3]
                        sim.addDrawingObjectItem(lines,t)
                    end
                end
            end
        end
    end
    
    -- if distances were processed, then send both unprojected angles
    -- and distances through the API.
    if #dists>0 then
        sim.setBufferProperty(sim.handle_scene, 'signal.hokuyo_range_data', sim.packFloatTable(dists))
        sim.setBufferProperty(sim.handle_scene, 'signal.hokuyo_angle_data', sim.packFloatTable(angles))
    end
    
    -- this flag marks that one execution
    -- already ended
    notFirstHere=true
end 