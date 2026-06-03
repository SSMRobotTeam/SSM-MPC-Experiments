function [outputArg1,outputArg2] = ParaFitting_Actuator(F)
%PARAFITTING_ACTUATOR 此处显示有关此函数的摘要
%   此处显示详细说明

    current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
    timestamp = char(current_time);  % 将 datetime 转换为字符串
    % 构造消息数据
    message = struct();
    message.command = 'set_motors';
    message.timestamp = timestamp;
    
    message.motorsCount = 5;
    message.isCtlModeUnified = 1;
    
    message.param.ctlMode = [3,3,3,3,3];
    message.param.posiSp  = [0,posi(3),posi(4),posi(1),posi(2)]*factor;
    message.param.forceSp = [0,0,0,0,0];
    
    % 编码为 JSON 字符串
    jsonStr = jsonencode(message);
    
    % 发送消息（带话题）
    topic = '/controller/setActuatorDate';
    socket.send(topic, org.zeromq.ZMQ.SNDMORE);  % 发送话题
    socket.send(jsonStr);  % 发送消息
    
    disp(['Sent @ ' message.timestamp]);
    disp(['拉绳长度： ', num2str(posi)]);
    % disp(['拉绳速度： ', num2str(veloc)]);

end

