% plot good-looking irfs and compare
close all
% shoch shock to plot? Choose i:
iShock  = 4;
cilow   = 0.68;
cihigh  = 0.90;

for iRes = n : -1 : 1
        PeterIRF{iRes, iShock}.irf(:,1) = prctile(struct_irf_record{iRes, iShock}(:, :), 100*(1-cihigh)/2);
        PeterIRF{iRes, iShock}.irf(:,2) = prctile(struct_irf_record{iRes, iShock}(:, :), 100*(1-cilow)/2);
        PeterIRF{iRes, iShock}.irf(:,3) = prctile(struct_irf_record{iRes, iShock}(:, :), 50);
        PeterIRF{iRes, iShock}.irf(:,4) = prctile(struct_irf_record{iRes, iShock}(:, :), 100*(1-(1-cilow)/2));
        PeterIRF{iRes, iShock}.irf(:,5) = prctile(struct_irf_record{iRes, iShock}(:, :), 100*(1-(1-cihigh)/2));
         
end

shock = {'A', 'B', 'C', 'D', 'E', 'F'};

    %for iRes = 1 : n
        subplot(2, 2, 1 )
        fill90 = [(1 : IRFperiods)', PeterIRF{1, iShock}.irf(:, 1); (IRFperiods : -1 : 1)', PeterIRF{1, iShock}.irf(end : -1 : 1, 5)  ];
        fill68 = [(1 : IRFperiods)', PeterIRF{1, iShock}.irf(:, 2); (IRFperiods : -1 : 1)', PeterIRF{1, iShock}.irf(end : -1 : 1, 4)  ];
        patch(fill90(:,1)    , fill90(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .2);    hold on
        patch(fill68(:,1)    , fill68(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .3);    hold on
        plot(PeterIRF{1, iShock}.irf(:, 3), '-','LineWidth', 1.9,'Color',[0.2 0.2 0.2])
        box on
        grid on
        set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.2, 'XLIM', [-inf inf])
        yline(0,'k-', 'LineWidth', 1);
        title('Government Expenditure','FontSize',12);
        ylabel('pp','FontSize',10);

        subplot(2, 2, 2 )
        fill90 = [(1 : IRFperiods)', PeterIRF{2, iShock}.irf(:, 1); (IRFperiods : -1 : 1)', PeterIRF{2, iShock}.irf(end : -1 : 1, 5)  ];
        fill68 = [(1 : IRFperiods)', PeterIRF{2, iShock}.irf(:, 2); (IRFperiods : -1 : 1)', PeterIRF{2, iShock}.irf(end : -1 : 1, 4)  ];
        patch(fill90(:,1)    , fill90(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .2);    hold on
        patch(fill68(:,1)    , fill68(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .3);    hold on
        plot(PeterIRF{2, iShock}.irf(:, 3), '-','LineWidth', 1.9,'Color',[0.2 0.2 0.2])
        box on
        grid on
        set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.2, 'XLIM', [-inf inf])
        yline(0,'k-', 'LineWidth', 1);
        title('GDP','FontSize',12);
        ylabel('pp','FontSize',10);


        subplot(2, 2, 3 )
        fill90 = [(1 : IRFperiods)', PeterIRF{3, iShock}.irf(:, 1); (IRFperiods : -1 : 1)', PeterIRF{3, iShock}.irf(end : -1 : 1, 5)  ];
        fill68 = [(1 : IRFperiods)', PeterIRF{3, iShock}.irf(:, 2); (IRFperiods : -1 : 1)', PeterIRF{3, iShock}.irf(end : -1 : 1, 4)  ];
        patch(fill90(:,1)    , fill90(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .2);    hold on
        patch(fill68(:,1)    , fill68(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .3);    hold on
        plot(PeterIRF{3, iShock}.irf(:, 3), '-','LineWidth', 1.9,'Color',[0.2 0.2 0.2])
        box on
        grid on
        set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.2, 'XLIM', [-inf inf])
        yline(0,'k-', 'LineWidth', 1);
        title('Consumption','FontSize',12);
        ylabel('pp','FontSize',10);


        subplot(2, 2, 4 )
        fill90 = [(1 : IRFperiods)', PeterIRF{4, iShock}.irf(:, 1); (IRFperiods : -1 : 1)', PeterIRF{4, iShock}.irf(end : -1 : 1, 5)  ];
        fill68 = [(1 : IRFperiods)', PeterIRF{4, iShock}.irf(:, 2); (IRFperiods : -1 : 1)', PeterIRF{4, iShock}.irf(end : -1 : 1, 4)  ];
        patch(fill90(:,1)    , fill90(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .2);    hold on
        patch(fill68(:,1)    , fill68(:,2)    , [0.122 0.467 0.706],'edgecolor',  'none', 'edgealpha', 0.3,  'facealpha', .3);    hold on
        plot(PeterIRF{4, iShock}.irf(:, 3), '-','LineWidth', 1.9,'Color',[0.2 0.2 0.2])
        box on
        grid on
        set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.2, 'XLIM', [-inf inf])
        yline(0,'k-', 'LineWidth', 1);
        title('Sentiment: Bundestag','FontSize',12);
        ylabel('std. dev.','FontSize',10);

% formatting with export function:
   %     width 18, height 15
    %end


