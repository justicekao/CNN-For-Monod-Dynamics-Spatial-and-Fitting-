% Save this file exactly as FitMicrobialSystemApp.m
function FitMicrobialSystemApp
    % Instantiate stable wide container framework layout natively
    fig = uifigure('Name', 'Multi-Strain Full Kinetics Matrix Fitter', 'Position', [50 50 1450 880]);
    W_side = 380;
    
    % Core global variables registry structure field matching setup tracks
    fields = {'w_r','w_k','w_P','w_K','w_c','w_Z','w_s','w_delta','w_m','w_D','w_m_tox','w_d_tox'};
    defaults = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]; 
    
    appData.fieldsList = fields;
    appData.sliders = struct();
    
    for i = 1:length(fields)
        appData.(fields{i}) = defaults(i);
    end
    
    % Main Panel Structural Splitting Containers
    mL = uipanel(fig, 'Position', [15 15 W_side 850]);
    mR = uipanel(fig, 'Position', [W_side+30 15 1450-W_side-45 850], 'BorderType', 'none');
    
    % Left Controller Tab Navigation Containers
    ctrlTabGroup = uitabgroup(mL, 'Position', [0 0 W_side 850]);
    tabSparsity = uitab(ctrlTabGroup, 'Title', 'Sparsity Config'); 
    tabICs = uitab(ctrlTabGroup, 'Title', 'Initial Conditions');
    
    % Core Operational Command Action Buttons Layout
    uibutton(tabSparsity, 'Text', 'Load CSV Data File', 'Position', [20 765 165 35], ...
        'ButtonPushedFcn', @(b,e) loadDataFile(fig));
    uibutton(tabSparsity, 'Text', 'Run Fit Optimization', 'Position', [195 765 165 35], ...
        'ButtonPushedFcn', @(b,e) runFitAndPlot(fig), 'BackgroundColor', [0.85 0.93 1.0], 'FontWeight', 'bold');
    
    uilabel(tabSparsity, 'Text', 'Paste Data Text Block Here:', 'Position', [20 730 340 22], 'FontWeight', 'bold');
    txtPasteArea = uitextarea(tabSparsity, 'Position', [20 585 340 140]);
    uibutton(tabSparsity, 'Text', 'Load Pasted Data Text Block', 'Position', [20 540 340 35], ...
        'ButtonPushedFcn', @(b,e) parsePastedTextData(fig));
        
    dropFitMode = uidropdown(tabSparsity, 'Position', [20 495 340 25], ...
        'Items', {'Isolate Selected Dataset Only', 'Jointly Across All Data'});
    
    % --- CONSTRAINED RE-ENGINEERING SEPARATION HOOKS ---
    % Split structural dimension hidden controllers cleanly to differentiate solutes from toxic compounds
    uilabel(tabSparsity, 'Text', 'Hidden Dimensions Config (Strains / Metab / Tox):', 'Position', [20 465 340 22], 'FontWeight', 'bold');
    spin_h_str = uispinner(tabSparsity, 'Position', [20 440 100 25], 'Limits', [0 5], 'Value', 0, ...
        'ValueChangedFcn', @(s,e) updateHiddenDim(fig, 'n_hidden_strains', s.Value));
    spin_h_met = uispinner(tabSparsity, 'Position', [135 440 110 25], 'Limits', [0 5], 'Value', 0, ...
        'ValueChangedFcn', @(s,e) updateHiddenDim(fig, 'n_hidden_metabolites', s.Value));
    spin_h_tox = uispinner(tabSparsity, 'Position', [260 440 100 25], 'Limits', [0 5], 'Value', 0, ...
        'ValueChangedFcn', @(s,e) updateHiddenDim(fig, 'n_hidden_toxins', s.Value));
        
    % Proceed dynamically to Part 2 to build the scroll layout sliders panel...
    % Part 2: Continuing inside the primary main App canvas creation routine shell loop...
    sp = uipanel(tabSparsity, 'Position', [10 10 360 420], 'Scrollable', 'on', 'BorderType', 'none');
    
    uilabel(sp, 'Text', 'Active Tube Parameter Tuning Focus Panel:', 'Position', [10 1055 320 22], 'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    dropParamTarget = uidropdown(sp, 'Position', [10 1030 320 25], 'Items', {'Tube Run 1'}, 'ItemsData', 1, 'ValueChangedFcn', @(d,e) handleParamTargetSwitch(fig, d.Value));

    labels = {'w - Growth (r)', 'w - Saturation (k)', 'w - Toxin Kill (P)', 'w - Tox Saturation (K)', 'w - Consumption (c)', 'w - Conjugation (Z)', 'w - Secretion (s)', 'w - Mortality (\delta)', 'w - Supply (m)', 'w - Dilution (D)', 'w - Tox Supply (m_tox)', 'w - Tox Decay (d_tox)'};
    
    for i = 1:length(labels)
        is2 = i > 6; rIdx = i - 6 * is2; x = 5 + 175 * is2; y = 940 - (rIdx * 78);
        uilabel(sp, 'Text', labels{i}, 'Position', [x, y+22, 155, 22], 'FontSize', 10);
        sld = uislider(sp, 'Position', [x, y, 155, 3], 'Limits', [0 12], 'Value', defaults(i), 'ValueChangedFcn', @(s,e) handleSliderMove(fig, fields{i}, s.Value, s));
        appData.sliders.(fields{i}) = sld;
    end

    % Standard independent dedicated error optimization tolerance limit target track slider layout bar
    uilabel(sp, 'Text', 'Controllable Error Tolerance (MSE Target):', 'Position', [10 70 260 22], 'FontWeight', 'bold');
    sld_tol = uislider(sp, 'Position', [10 50, 320, 3], 'Limits', [0, 0.1], 'Value', 0.01);
    appData.sld_tol = sld_tol;

    uilabel(sp, 'Text', 'Alternative IC Scale Factor Knob:', 'Position', [10 20 200 22]);
    sld_ic = uislider(sp, 'Position', [170 30, 160, 3], 'Limits', [0.1, 5.0], 'Value', 1.0, 'ValueChangedFcn', @(s,e) updateICScaleInstant(fig, s.Value));

    % Right plot subplots window grids shell properties parameters tracking layout assignments setup
    topCtrlBar = uipanel(mR, 'Position', [0 780 1450-W_side-45 70], 'BorderType', 'none');
    uilabel(topCtrlBar, 'Text', 'Display Target View:', 'Position', [10 25 120 22], 'FontWeight', 'bold');
    dropDatasetData = uidropdown(topCtrlBar, 'Position', [140 24 220 25], 'Items', {'Tube Run 1'}, 'ItemsData', 1, 'ValueChangedFcn', @(d,e) syncAndApplyUIPriority(fig, d.Value));
    
    chkShowHiddenStr = uicheckbox(topCtrlBar, 'Text', 'Show Hidden Strains', 'Position', [380 25 150 22], 'Value', false, 'ValueChangedFcn', @(c,e) triggerFastReplot(fig));
    chkShowHiddenSub = uicheckbox(topCtrlBar, 'Text', 'Show Hidden Metabolites', 'Position', [540 25 170 22], 'Value', false, 'ValueChangedFcn', @(c,e) triggerFastReplot(fig));
    chkShowHiddenTox = uicheckbox(topCtrlBar, 'Text', 'Show Hidden Toxins', 'Position', [720 25 150 22], 'Value', false, 'ValueChangedFcn', @(c,e) triggerFastReplot(fig));
    
    tgR = uitabgroup(mR, 'Position', [0 0 1450-W_side-45 770]);
    t1 = uitab(tgR, 'Title', 'Data vs Fit Prediction Curves'); ax1 = uiaxes(t1, 'Position', [40 40 1000 660]);
    t2 = uitab(tgR, 'Title', 'Alternative ICMod Trajectories'); ax2 = uiaxes(t2, 'Position', [40 40 1000 660]);
    t3 = uitab(tgR, 'Title', 'Parameter Heatmap Matrix Explorer');
    
    dropMatrix = uidropdown(t3, 'Position', [280 710 250 25], 'Items', labels, 'ItemsData', 1:12, 'Value', 1, 'ValueChangedFcn', @(d,e) triggerFastReplot(fig));
    gridPanel = uipanel(t3, 'Position', [20 20 1020 660]);
    icScrollPanel = uipanel(tabICs, 'Position', [5 5 W_side-15 820], 'Scrollable', 'on');
    
    appData.ic_scale = 1.0; appData.ax1 = ax1; appData.ax2 = ax2; appData.gridPanel = gridPanel; appData.dropMatrix = dropMatrix; 
    appData.txtPasteArea = txtPasteArea; appData.dropDatasetData = dropDatasetData; appData.dropFitMode = dropFitMode;
    appData.spin_h_str = spin_h_str; appData.spin_h_met = spin_h_met; appData.spin_h_tox = spin_h_tox; appData.dropParamTarget = dropParamTarget;
    appData.chkShowHiddenStr = chkShowHiddenStr; appData.chkShowHiddenSub = chkShowHiddenSub; appData.chkShowHiddenTox = chkShowHiddenTox;
    appData.icScrollPanel = icScrollPanel; appData.hasOptimized = false; 
    
    fig.UserData = appData;
    loadDefaultPublicationDataset(fig);
end
function loadDefaultPublicationDataset(fig)
    % Complete publication sample configuration. Contains mixed normal tubes and community tubes
    % tracking explicit target line-level parameter locks and specific segmented dimension limits.
    defaultText = [...
        '# DatasetID: 1' newline ...
        '# HiddenStrains: 0' newline ...
        '# HiddenMetabolites: 0' newline ...
        '# HiddenToxins: 0' newline ...
        'Time,Recipients,Donors,Transconjugants,Nutrient_1,Toxin_1' newline ...
        '0,5.00e+07,1.00e+07,1.00e+00,10.00,0.01' newline ...
        '4,1.05e+08,1.90e+07,6.20e+03,7.20,0.07' newline ...
        '8,2.10e+08,3.50e+07,1.15e+06,3.80,0.32' newline ...
        '12,3.70e+08,5.80e+07,1.20e+08,1.50,1.20' newline ...
        '16,4.90e+08,7.50e+07,4.10e+08,0.55,3.40' newline ...
        '20,5.15e+08,8.10e+08,5.30e+08,0.20,6.10' newline ...
        '24,5.20e+08,8.25e+08,5.45e+08,0.08,8.00' newline ...
        '# DatasetID: 2' newline ...
        '# HiddenStrains: 0' newline ...
        '# HiddenMetabolites: 1' newline ...
        '# HiddenToxins: 0' newline ...
        '# MatrixFreeze: r(1,1)=0.45' newline ...
        'Time,Recipients,Donors,Transconjugants,Nutrient_1,Toxin_1' newline ...
        '0,2.00e+07,4.00e+07,1.00e+00,12.00,1.00e-4' newline ...
        '4,5.50e+07,6.20e+07,2.10e+04,9.10,0.05' newline ...
        '8,1.40e+08,9.50e+07,4.80e+06,5.40,0.22' newline ...
        '12,2.90e+08,1.50e+08,6.50e+07,2.10,0.95' newline ...
        '16,3.80e+08,2.10e+08,1.90e+08,0.75,2.40' newline ...
        '20,4.10e+08,2.40e+08,2.90e+08,0.30,4.10' newline ...
        '24,4.25e+08,2.50e+08,3.20e+08,0.11,5.50'];
    
    appData = fig.UserData; 
    appData.txtPasteArea.Value = splitlines(defaultText); 
    fig.UserData = appData; 
    parsePastedTextData(fig);
end
function parsePastedTextData(fig)
    appData = fig.UserData; lines = appData.txtPasteArea.Value; 
    if isempty(lines) || (length(lines) == 1 && isempty(lines{1}))
        return; 
    end
    try
        tempCell = cellfun(@(x) [x, newline], lines, 'UniformOutput', false);
        raw = [tempCell{:}]; lns = splitlines(raw);
        staticFieldsFilter = {'w_r','w_k','w_P','w_K','w_c','w_Z','w_s','w_delta','w_m','w_D','w_m_tox','w_d_tox'};
        
        experimentBlocks = cell(20, 1); experimentMetas = cell(20, 1);
        currID = 1; blockLineCount = 0; experimentMetas{currID} = struct();
        experimentMetas{currID}.Freezes = {}; 
        
        for i = 1:length(lns)
            ln = strtrim(lns{i}); if isempty(ln); continue; end
            if startsWith(ln, '#')
                ln = strtrim(extractAfter(ln, '#'));
                if startsWith(ln, 'DatasetID:')
                    newID = str2double(strtrim(extractAfter(ln, ':')));
                    if ~isnan(newID) && newID ~= currID
                        currID = newID; blockLineCount = 0; 
                        experimentMetas{currID} = struct();
                        experimentMetas{currID}.Freezes = {};
                    end
                end
                if startsWith(ln, 'MatrixFreeze:')
                    freezeExpr = strtrim(extractAfter(ln, 'MatrixFreeze:'));
                    experimentMetas{currID}.Freezes{end+1} = freezeExpr;
                elseif contains(ln, ':')
                    k = strtrim(extractBefore(ln, ':')); v = strtrim(extractAfter(ln, ':'));
                    if ismember(k, [staticFieldsFilter, {'HiddenStrains','HiddenMetabolites','HiddenToxins'}])
                        experimentMetas{currID}.(k) = str2double(v); 
                    elseif strcmp(k, 'MatrixZeros'); experimentMetas{currID}.MatrixZeros = v; end
                end
            else
                blockLineCount = blockLineCount + 1;
                if blockLineCount == 1 && ~startsWith(ln, 'Time')
                    experimentBlocks{currID}{1} = 'Time,Strains,Recipients,Transconjugants,Substrate,Antibiotics';
                    blockLineCount = blockLineCount + 1;
                end
                experimentBlocks{currID}{blockLineCount} = ln;
            end
        end
        
        validIdxs = find(~cellfun(@isempty, experimentBlocks)); appData.datasets = cell(length(validIdxs), 1);
        for idx = 1:length(validIdxs)
            realID = validIdxs(idx); blockLines = experimentBlocks{realID};
            transformedCells = cellfun(@(x) [x, newline], blockLines, 'UniformOutput', false); blockText = [transformedCells{:}];
            fid = fopen('tmp_split.csv', 'w'); fprintf(fid, '%s', blockText); fclose(fid); 
            
            % --- RE-ENGINEERED: TYPE-SAFE IMPORT SCANNER ---
            % Treats the metadata label 'Strains' as text columns while locking remaining parameters as numeric [1]
            opts = detectImportOptions('tmp_split.csv');
            if ismember('Strains', opts.VariableNames)
                opts = setvartype(opts, 'Strains', 'string');
            end
            T = readtable('tmp_split.csv', opts); delete('tmp_split.csv'); 
            
            % Isolate text elements away into independent storage metadata boxes [1]
            if ismember('Strains', T.Properties.VariableNames)
                strainTags = string(T.Strains);
                T.Strains = []; % Removes character text to secure clean numeric conversions [1]
            else
                strainTags = repmat("Control_Tube", height(T), 1);
            end
            
            allCols = T.Properties.VariableNames;
            cols = allCols(~strcmp(allCols, 'Time') & ~strcmp(allCols, 'DatasetID'));
            
            dsMeta = experimentMetas{realID};
            for f = 1:length(staticFieldsFilter)
                if ~isfield(dsMeta, staticFieldsFilter{f}); dsMeta.(staticFieldsFilter{f}) = 4; end
            end
            if ~isfield(dsMeta, 'HiddenStrains'); dsMeta.HiddenStrains = 0; end
            if ~isfield(dsMeta, 'HiddenMetabolites'); dsMeta.HiddenMetabolites = 0; end
            if ~isfield(dsMeta, 'HiddenToxins'); dsMeta.HiddenToxins = 0; end
            
            % Pack pure, safe numeric array loops into your tracking memory workspace space [1]
            appData.datasets{idx} = struct('t', T.Time, 'Y', table2array(T(:, cols)), ...
                'names', {cols}, 'meta', dsMeta, 'strain_tags', strainTags);
        end
        appData.hasOptimized = false; fig.UserData = appData; updateDatasetDropdowns(fig); syncAndApplyUIPriority(fig, 1);
    catch ME
        uialert(fig, ['Parsing failure: ', ME.message], 'Error');
    end
end

function syncAndApplyUIPriority(fig, dsIdx)
    appData = fig.UserData; if dsIdx > length(appData.datasets); return; end
    ds = appData.datasets{dsIdx}; fields = appData.fieldsList; appData.dropParamTarget.Value = dsIdx;
    
    for i = 1:length(fields)
        appData.sliders.(fields{i}).Value = ds.meta.(fields{i});
        appData.sliders.(fields{i}).Enable = 'on'; 
    end
    appData.spin_h_str.Enable = 'on'; 
    appData.spin_h_sub.Enable = 'on';
    if isfield(appData, 'spin_h_tox'); appData.spin_h_tox.Enable = 'on'; end
    
    if ds.meta.HiddenStrains > 0; appData.spin_h_str.Enable = 'off'; end
    if ds.meta.HiddenMetabolites > 0; appData.spin_h_sub.Enable = 'off'; end
    if isfield(ds.meta, 'HiddenToxins') && ds.meta.HiddenToxins > 0 && isfield(appData, 'spin_h_tox')
        appData.spin_h_tox.Enable = 'off'; 
    end
    
    if isfield(ds.meta, 'Freezes') && ~isempty(ds.meta.Freezes)
        for f = 1:length(ds.meta.Freezes)
            expr = ds.meta.Freezes{f}; if isempty(expr); continue; end
            if contains(expr, '(')
                mToken = strtrim(extractBefore(expr, '('));
                targetSliderName = ['w_' mToken];
                if isfield(appData.sliders, targetSliderName)
                    appData.sliders.(targetSliderName).Enable = 'off';
                end
            end
        end
    end
    
    appData.spin_h_str.Value = ds.meta.HiddenStrains; appData.n_hidden_strains = ds.meta.HiddenStrains;
    appData.spin_h_sub.Value = ds.meta.HiddenMetabolites; appData.n_hidden_solutes = ds.meta.HiddenMetabolites;
    if isfield(appData, 'spin_h_tox') && isfield(ds.meta, 'HiddenToxins')
        appData.spin_h_tox.Value = ds.meta.HiddenToxins; appData.n_hidden_toxins = ds.meta.HiddenToxins;
    end
    fig.UserData = appData; refreshICPanelControls(fig); triggerFastReplot(fig);
end

function handleParamTargetSwitch(fig, focusedIdx)
    appData = fig.UserData; appData.dropDatasetData.Value = focusedIdx; fig.UserData = appData; syncAndApplyUIPriority(fig, focusedIdx);
end

function handleSliderMove(fig, fieldName, value, sldObj)
    sldObj.Value = round(value); appData = fig.UserData; appData.(fieldName) = round(value);
    appData.datasets{appData.dropParamTarget.Value}.meta.(fieldName) = round(value); fig.UserData = appData; triggerFastReplot(fig);
end
function runFitAndPlot(fig)
    appData = fig.UserData; totalVars = size(appData.datasets{1}.Y, 2); lbls = appData.datasets{1}.names;
    
    validLabels = lbls(~strcmpi(lbls, 'Strains'));
    S_obs = sum(contains(validLabels,'Recipient') | contains(validLabels,'Donor') | contains(validLabels,'Transconjugant') | contains(validLabels,'Strain'));
    M_obs = sum(contains(validLabels,'Nutrient') | contains(validLabels,'Substrate') | contains(validLabels,'Metabolite')); 
    
    % --- FIXED REGEX LOOKUP: Added 'Antibiotic' to capture your new header columns natively ---
    T_obs = sum(contains(validLabels,'Toxin') | contains(validLabels,'Inhibitor') | contains(validLabels,'Antibiotic'));
    if S_obs==0; S_obs=3; M_obs=1; T_obs=1; end 
    
    isJoint = strcmp(appData.dropFitMode.Value, 'Jointly Across All Data');
    d = uiprogressdlg(fig, 'Title', 'Running Global SOR Fitter', ...
        'Message', 'Launching multi-start parallel gradient paths...', 'Indeterminate', 'off', 'Value', 0, 'Cancelable', 'on');
    
    maxS_hid = 0; maxM_hid = 0; maxT_hid = 0;
    for i = 1:length(appData.datasets)
        maxS_hid = max(maxS_hid, appData.datasets{i}.meta.HiddenStrains);
        maxM_hid = max(maxM_hid, appData.datasets{i}.meta.HiddenMetabolites);
        if isfield(appData.datasets{i}.meta, 'HiddenToxins')
            maxT_hid = max(maxT_hid, appData.datasets{i}.meta.HiddenToxins);
        end
    end
    S = S_obs + maxS_hid; M = M_obs + maxM_hid; T = T_obs + maxT_hid;
    
    globalSz = [S*M, S*M, S*T, S*T, S*M, S*S, S*T, S, S, M, M, T, T, T];
    numDatasets = length(appData.datasets);
    localICSizePerDataset = maxS_hid + maxM_hid + maxT_hid;
    
    if isJoint
        sz = [globalSz, repmat(localICSizePerDataset, 1, numDatasets)];
    else
        sz = [globalSz, localICSizePerDataset];
    end
    
    tot = sum(sz); c = cumsum([0, sz]);
    w_vec = zeros(1, length(appData.fieldsList));
    for i = 1:length(appData.fieldsList); w_vec(i) = appData.datasets{appData.dropParamTarget.Value}.meta.(appData.fieldsList{i}); end
    
    p0 = -2.5 * ones(tot, 1); p0(c(1)+1:c(2)) = -0.5; p0(c(2)+1:c(3)) = 1.0; 
    
    if isJoint
        for i = 1:numDatasets
            idxBlockStart = c(14 + i); 
            if maxS_hid > 0; p0(idxBlockStart+1 : idxBlockStart+maxS_hid) = 4.0; end
            if maxM_hid > 0; p0(idxBlockStart+maxS_hid+1 : idxBlockStart+maxS_hid+maxM_hid) = 1.0; end
            if maxT_hid > 0; p0(idxBlockStart+maxS_hid+maxM_hid+1 : idxBlockStart+maxS_hid+maxM_hid+maxT_hid) = -1.0; end
        end
    else
        idxBlockStart = c(15);
        if maxS_hid > 0; p0(idxBlockStart+1 : idxBlockStart+maxS_hid) = 4.0; end
        if maxM_hid > 0; p0(idxBlockStart+maxS_hid+1 : idxBlockStart+maxS_hid+maxM_hid) = 1.0; end
        if maxT_hid > 0; p0(idxBlockStart+maxS_hid+maxM_hid+1 : idxBlockStart+maxS_hid+maxM_hid+maxT_hid) = -1.0; end
    end
    p0(isnan(p0)|isinf(p0)) = -2.5;
    lb = -12.0 * ones(tot, 1); ub = 4.0 * ones(tot, 1);
    
    icProfile.obs_y0 = zeros(totalVars, 1);
    for i = 1:totalVars
        if isfield(appData, 'ic_inputs') && isfield(appData.ic_inputs, sprintf('obs_%d', i))
            icProfile.obs_y0(i) = appData.ic_inputs.(sprintf('obs_%d', i)).Value;
        else; icProfile.obs_y0(i) = appData.datasets{1}.Y(1, i); end
    end
    
    if isJoint
        obj = @(p) computeJointResidual(p, appData.datasets, S_obs, M_obs, T_obs, S, M, T, sz, icProfile, w_vec);
        opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'none', ...
            'MaxIterations', 80, 'FunctionTolerance', 1e-4, 'FiniteDifferenceStepSize', 1e-2, 'UseParallel', true);
        
        maxCycles = 15; currentCycle = 1; bestScore = Inf; pBest = p0;
        while currentCycle <= maxCycles
            drawnow; if d.CancelRequested; close(d); return; end
            d.Value = currentCycle / maxCycles; d.Message = sprintf('Global Joint Pass: Cycle %d/%d (Best MSE: %.5f)', currentCycle, maxCycles, bestScore);
            pRaw = lsqnonlin(obj, p0, lb, ub, opts); score = mean(obj(pRaw).^2);
            if score < bestScore; bestScore = score; pBest = pRaw; end
            if score <= appData.sld_tol.Value; break; end
            p0 = pBest + (-1.5 + 3.0*rand(tot,1)); currentCycle = currentCycle + 1;
        end
        
        for i = 1:length(appData.datasets)
            pCustomLocal = [pBest(1:c(15)); pBest(c(14+i)+1 : c(15+i))];
            localSz = [globalSz, localICSizePerDataset];
            appData.local_fits{i} = struct('pRaw', pCustomLocal, 'sizes', localSz, 'S', S, 'M', M, 'T', T, 'icProfile', icProfile);
        end
    else
        sIdx = appData.dropParamTarget.Value;
        obj = @(p) computeJointResidual(p, appData.datasets(sIdx), S_obs, M_obs, T_obs, S, M, T, sz, icProfile, w_vec);
        opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'none', ...
            'MaxIterations', 80, 'FunctionTolerance', 1e-4, 'FiniteDifferenceStepSize', 1e-2, 'UseParallel', true);
        
        maxCycles = 15; currentCycle = 1; bestScore = Inf; pBest = p0;
        while currentCycle <= maxCycles
            drawnow; if d.CancelRequested; close(d); return; end
            d.Value = currentCycle / maxCycles; d.Message = sprintf('Isolated Pass Tube %d: Cycle %d/%d', sIdx, currentCycle, maxCycles);
            pRaw = lsqnonlin(obj, p0, lb, ub, opts); score = mean(obj(pRaw).^2);
            if score < bestScore; bestScore = score; pBest = pRaw; end
            if score <= appData.sld_tol.Value; break; end
            p0 = pBest + (-1.5 + 3.0*rand(tot,1)); currentCycle = currentCycle + 1;
        end
        appData.local_fits{sIdx} = struct('pRaw', pBest, 'sizes', sz, 'S', S, 'M', M, 'T', T, 'icProfile', icProfile);
    end
    close(d); appData.hasOptimized = true; fig.UserData = appData; triggerFastReplot(fig);
end


function loadDataFile(fig)
    [file, path] = uigetfile('*.csv'); if isequal(file, 0); return; end
    T = readtable(fullfile(path, file)); appData = fig.UserData; T.DatasetID = cumsum(T.Time == 0 | (1:height(T))' == 1); ids = unique(T.DatasetID); appData.datasets = cell(length(ids), 1);
    for i = 1:length(ids)
        subT = T(T.DatasetID == ids(i), :); cols = setdiff(subT.Properties.VariableNames, {'Time', 'DatasetID'});
        dsMeta = struct(); for f = 1:length(appData.fieldsList); dsMeta.(appData.fieldsList{f}) = 4; end
        dsMeta.HiddenStrains = 0; 
        dsMeta.HiddenMetabolites = 0; 
        dsMeta.HiddenToxins = 0; 
        dsMeta.Freezes = {};
        appData.datasets{i} = struct('t', subT.Time, 'Y', table2array(subT(:, cols)), 'names', {cols}, 'meta', dsMeta);
    end
    appData.hasOptimized = false; fig.UserData = appData; updateDatasetDropdowns(fig); syncAndApplyUIPriority(fig, 1);
end

function updateHiddenDim(fig, field, val)
    appData = fig.UserData; 
    if strcmp(field,'n_hidden_strains')
        tagStr = 'HiddenStrains'; 
    elseif strcmp(field,'n_hidden_solutes')
        tagStr = 'HiddenMetabolites'; 
    else
        tagStr = 'HiddenToxins'; 
    end
    appData.datasets{appData.dropParamTarget.Value}.meta.(tagStr) = val; appData.(field) = val; fig.UserData = appData; triggerFastReplot(fig);
end

function updateICScaleInstant(fig, val)
    appData = fig.UserData; appData.ic_scale = val; fig.UserData = appData; triggerFastReplot(fig);
end


function refreshICPanelControls(fig)
    appData = fig.UserData; panel = appData.icScrollPanel; delete(panel.Children);
    totalVars = size(appData.datasets{1}.Y, 2); names = appData.datasets{1}.names; yc = 1000; 
    uilabel(panel, 'Text', 'USER INITIAL CONDITIONS', 'Position', [10, yc-25, 250, 22], 'FontWeight', 'bold'); yc = yc - 35; appData.ic_inputs = struct();
    for i = 1:totalVars
        uilabel(panel, 'Text', names{i}, 'Position', [10, yc, 140, 22]);
        appData.ic_inputs.(sprintf('obs_%d', i)) = uieditfield(panel, 'numeric', 'Position', [160, yc, 80, 22], 'Value', appData.datasets{1}.Y(1, i)); yc = yc - 30;
    end
    fig.UserData = appData;
end

function updateDatasetDropdowns(fig)
    appData = fig.UserData; n = length(appData.datasets); 
    lbls = arrayfun(@(x) sprintf('Tube Run %d', x), 1:n, 'UniformOutput', false);
    appData.dropDatasetData.Items = lbls; appData.dropDatasetData.ItemsData = 1:n; appData.dropDatasetData.Value = 1;
    appData.dropParamTarget.Items = lbls; appData.dropParamTarget.ItemsData = 1:n; appData.dropParamTarget.Value = 1;
    fig.UserData = appData;
end
function finalRes = computeJointResidual(p, datasets, S_obs, M_obs, T_obs, S, M, T, sz, icProfile, w_vec)
    globalParamsCount = sum(sz(1:14));
    pGlobalSegment = p(1:globalParamsCount);
    idx = 0;
    for i = 1:14
        segment = pGlobalSegment(idx+1 : idx+sz(i));
        if sz(i) > 1
            flat = sort(abs(segment), 'descend'); w_c = max(1, w_vec(i)); thresh = flat(min(w_c, length(flat)));
            if w_vec(i) == 0; thresh = Inf; end; segment(abs(segment) < thresh) = 0; pGlobalSegment(idx+1 : idx+sz(i)) = segment;
        end
        idx = idx + sz(i);
    end
    p(1:globalParamsCount) = pGlobalSegment;
    
    finalRes = [];
    isMultipleDatasetsInjected = length(datasets) > 1;
    c = cumsum([0, sz]); 
    
    for dIdx = 1:length(datasets)
        ds = datasets{dIdx};
        
        if isMultipleDatasetsInjected
            pLocalAssembled = [p(1:globalParamsCount); p(c(14+dIdx)+1 : c(15+dIdx))];
            localSizingLayout = [sz(1:14), sz(14+dIdx)];
        else
            pLocalAssembled = p;
            localSizingLayout = sz;
        end
        
        [~, pStrLocal] = unpackAllMatrices(pLocalAssembled, S, M, T, localSizingLayout, icProfile, {ds}, S_obs, M_obs, T_obs); 
        pStrLocal = applyZeroLocationConstraints(pStrLocal, ds);
        
        if isfield(ds.meta, 'Freezes') && ~isempty(ds.meta.Freezes)
            for f = 1:length(ds.meta.Freezes)
                expr = ds.meta.Freezes{f}; if isempty(expr) || ~contains(expr,'('); continue; end
                try
                    mName = strtrim(extractBefore(expr, '('));
                    r = str2double(extractBetween(expr, '(', ',')); c = str2double(extractBetween(expr, ',', ')'));
                    val = str2double(extractAfter(expr, '='));
                    if isfield(pStrLocal, mName); pStrLocal.(mName)(r,c) = val; end
                catch
                end
            end
        end
        
        y0 = zeros(S+M+T, 1);
        for i = 1:S_obs; if i<=length(icProfile.obs_y0); y0(i)=log(max(1e-6, icProfile.obs_y0(i)/1e8)); else; y0(i)=log(1e-6); end; end
        if S > S_obs; y0(S_obs+1:S) = log(max(1e-6, pStrLocal.hidden_ICs{1}.strains(1:(S-S_obs)))); end
        for i = 1:M_obs; src=S_obs+i; if src<=length(icProfile.obs_y0); y0(S+i)=log(max(1e-6, icProfile.obs_y0(src))); else; y0(S+i)=log(1e-2); end; end
        if M > M_obs; y0(S+M_obs+1:S+M) = log(max(1e-6, pStrLocal.hidden_ICs{1}.metabolites(1:(M-M_obs)))); end
        for i = 1:T_obs; src=S_obs+M_obs+i; if src<=length(icProfile.obs_y0); y0(S+M+i)=log(max(1e-6, icProfile.obs_y0(src))); else; y0(S+M+i)=log(1e-3); end; end
        if T > T_obs; y0(S+M+T_obs+1:end) = log(max(1e-6, pStrLocal.hidden_ICs{1}.toxins(1:(T-T_obs)))); end
        
        [~, ySim] = ode23s(@(t,y) explicitSystemODE(t, y, pStrLocal), ds.t, y0, odeset('AbsTol',1e-5,'RelTol',1e-4));
        if size(ySim, 1) == length(ds.t) && all(isfinite(ySim(:)))
            yObs = [exp(ySim(:, 1:S_obs))*1e8, exp(ySim(:, S+1:S+M_obs)), exp(ySim(:, S+M+1:S+M+T_obs))];
            logSim = log10(max(1, yObs)); 
            
            % --- FIXED VECTOR SLICING PROFILE: Ensure data columns match simulation matrix widths precisely ---
            trimmedYData = ds.Y(:, 1:(S_obs+M_obs+T_obs));
            logData = log10(max(1, trimmedYData));
            
            maxLog = max(logData, [], 1); minLog = min(logData, [], 1); rangeLog = maxLog - minLog; rangeLog(rangeLog < 1e-2) = 1.0;
            err = (logSim - logData) ./ repmat(rangeLog, length(ds.t), 1); finalRes = [finalRes; err(:)];
        else
            finalRes = [finalRes; 15.0 * ones(length(ds.t) * (S_obs+M_obs+T_obs), 1)];
        end
    end
end


function pStr = applyZeroLocationConstraints(pStr, ds)
    if isfield(ds.meta, 'MatrixZeros')
        exprs = split(ds.meta.MatrixZeros, ',');
        for e = 1:length(exprs)
            ex = strtrim(exprs{e}); if isempty(ex); continue; end
            try
                mName = extractBefore(ex, '('); idxs = str2num(char(extractBetween(ex, '(', ')')));
                if isfield(pStr, mName); pStr.(mName)(idxs(1), idxs(2)) = 0; end
            catch
            end
        end
    end
end

function [Mats, pStr] = unpackAllMatrices(p, S, M, T, sz, icP, datasets, S_obs, M_obs, T_obs)
    c = cumsum([0, sz]);
    % Ensure parameter tracking blocks are cleanly extracted regardless of shape orient configurations
    p = p(:); 
    
    Mats.r = 10.^(reshape(p(c(1)+1:c(2)), S, M) - 1);       
    Mats.k = 10.^(reshape(p(c(2)+1:c(3)), S, M));
    Mats.P = 10.^(reshape(p(c(3)+1:c(4)), S, T) - 1);       
    Mats.K = 10.^(reshape(p(c(4)+1:c(5)), S, T));
    Mats.c_consumption = 10.^(reshape(p(c(5)+1:c(6)), S, M) - 1); 
    Mats.Z_transconjugation = 10.^(reshape(p(c(6)+1:c(7)), S, S) - 2); 
    Mats.s_secretion = 10.^(reshape(p(c(7)+1:c(8)), S, T) - 1);   
    
    % --- FIXED EXPLICIT COLUMNS PASS ENGINE ---
    % Appending (:) structurally forces vectors to build as strict columns, eliminating dimension faults
    Mats.delta = 10.^(reshape(p(c(8)+1:c(9)), S, 1) - 2);         Mats.delta = Mats.delta(:);
    Mats.g = reshape(p(c(9)+1:c(10)), S, 1);                     Mats.g = Mats.g(:);
    Mats.m_supply = 10.^(reshape(p(c(10)+1:c(11)), M, 1));        Mats.m_supply = Mats.m_supply(:);
    Mats.D_dilution = 10.^(reshape(p(c(11)+1:c(12)), M, 1) - 1);  Mats.D_dilution = Mats.D_dilution(:);
    Mats.m_toxin_supply = 10.^(reshape(p(c(12)+1:c(13)), T, 1));  Mats.m_toxin_supply = Mats.m_toxin_supply(:);
    Mats.d_toxin_decay = 10.^(reshape(p(c(13)+1:c(14)), T, 1));   Mats.d_toxin_decay = Mats.d_toxin_decay(:);
    Mats.d_outflow = reshape(p(c(14)+1:c(15)), T, 1);             Mats.d_outflow = Mats.d_outflow(:);
    
    S_hid = S - S_obs; M_hid = M - M_obs; T_hid = T - T_obs;
    Mats.hidden_ICs = cell(length(datasets), 1);
    for i = 1:length(datasets)
        idxStart = c(15);
        if S_hid > 0
            Mats.hidden_ICs{i}.strains = 10.^p(idxStart+1 : idxStart+S_hid); idxStart = idxStart + S_hid;
            Mats.hidden_ICs{i}.strains = Mats.hidden_ICs{i}.strains(:);
        else; Mats.hidden_ICs{i}.strains = []; end
        if M_hid > 0
            Mats.hidden_ICs{i}.metabolites = 10.^p(idxStart+1 : idxStart+M_hid); idxStart = idxStart + M_hid;
            Mats.hidden_ICs{i}.metabolites = Mats.hidden_ICs{i}.metabolites(:);
        else; Mats.hidden_ICs{i}.metabolites = []; end
        if T_hid > 0
            Mats.hidden_ICs{i}.toxins = 10.^p(idxStart+1 : idxStart+T_hid);
            Mats.hidden_ICs{i}.toxins = Mats.hidden_ICs{i}.toxins(:);
        else; Mats.hidden_ICs{i}.toxins = []; end
    end
    pStr = Mats; pStr.S = S; pStr.M = M; pStr.T = T;
end

function dydt = explicitSystemODE(~, y, p)
    state = exp(y(:)); dydt = zeros(size(y)); 
    N = state(1:p.S); N = N(:);
    x = state(p.S+1 : p.S+p.M); x = x(:);
    tox = state(p.S+p.M+1 : end); tox = tox(:);
    
    X_s = repmat(x', p.S, 1) ./ (max(1e-4, p.k) + 1e-4); denom_x = 1 + sum(X_s, 2); 
    Y_s = repmat(tox', p.S, 1) ./ (max(1e-4, p.K) + 1e-4); denom_y = 1 + sum(Y_s, 2);
    
    conjugationInflow = zeros(p.S, 1);
    if p.S >= 3
        massActionRate = p.Z_transconjugation(3,1) * N(1) * (N(2) + N(3));
        conjugationInflow(1) = -massActionRate; conjugationInflow(3) = massActionRate;  
    end
    conjugationInflow = conjugationInflow(:);
    
    % Dimensions are now strictly forced to align, preventing evaluation failures
    dN = N .* (sum(p.r .* X_s, 2) ./ denom_x - sum(p.P .* Y_s, 2) ./ denom_y - p.delta) + conjugationInflow;
    dx = p.m_supply - p.D_dilution .* x - sum((p.c_consumption .* X_s ./ denom_x) .* repmat(N, 1, p.M), 1)';
    dtox = p.m_toxin_supply - p.d_toxin_decay .* tox + sum((p.s_secretion .* Y_s ./ denom_y) .* repmat(N, 1, p.T), 1)';
    
    dydt(1:p.S) = dN ./ N; dydt(p.S+1 : p.S+p.M) = dx ./ x; dydt(p.S+p.M+1 : end) = dtox ./ tox;
    dydt = dydt(:);
end

function triggerFastReplot(fig)
    appData = fig.UserData; lbls = appData.datasets{1}.names;
    
    validLabels = lbls(~strcmpi(lbls, 'Strains'));
    S_obs = sum(contains(validLabels,'Recipient') | contains(validLabels,'Donor') | contains(validLabels,'Transconjugant') | contains(validLabels,'Strain'));
    M_obs = sum(contains(validLabels,'Nutrient') | contains(validLabels,'Substrate') | contains(validLabels,'Metabolite')); 
    
    % --- FIXED REGEX LOOKUP MAPS ---
    T_obs = sum(contains(validLabels,'Toxin') | contains(validLabels,'Inhibitor') | contains(validLabels,'Antibiotic'));
    if S_obs==0; S_obs=3; M_obs=1; T_obs=1; end
    
    tDSIdx = appData.dropDatasetData.Value; ds = appData.datasets{tDSIdx};
    fields = appData.fieldsList; w_vec = zeros(1, length(fields)); 
    for i = 1:length(fields); w_vec(i) = ds.meta.(fields{i}); end
    
    maxS_hid = 0; maxM_hid = 0; maxT_hid = 0;
    for i = 1:length(appData.datasets)
        maxS_hid = max(maxS_hid, appData.datasets{i}.meta.HiddenStrains); 
        maxM_hid = max(maxM_hid, appData.datasets{i}.meta.HiddenMetabolites);
        if isfield(appData.datasets{i}.meta, 'HiddenToxins')
            maxT_hid = max(maxT_hid, appData.datasets{i}.meta.HiddenToxins);
        end
    end
    S = S_obs + maxS_hid; M = M_obs + maxM_hid; T = T_obs + maxT_hid;
    sz = [S*M, S*M, S*T, S*T, S*M, S*S, S*T, S, S, M, M, T, T, T, maxS_hid, maxM_hid, maxT_hid];
    
    isHasFit = isfield(appData, 'local_fits') && length(appData.local_fits) >= tDSIdx && ~isempty(appData.local_fits{tDSIdx});
    isJointMode = strcmp(appData.dropFitMode.Value, 'Jointly Across All Data');
    
    if isHasFit
        lf = appData.local_fits{tDSIdx}; pLive = lf.pRaw; sz = lf.sizes; S = lf.S; M = lf.M; T = lf.T; icP = lf.icProfile; 
    else
        icP.obs_y0 = ds.Y(1, :)'; pLive = -1.5 * ones(sum(sz), 1); pLive(1:S*M) = -0.5;
    end
    
    if isfield(ds.meta, 'Freezes') && ~isempty(ds.meta.Freezes)
        [~, pStrTmp] = unpackAllMatrices(pLive, S, M, T, sz, icP, {ds}, S_obs, M_obs, T_obs);
        for f = 1:length(ds.meta.Freezes)
            expr = ds.meta.Freezes{f}; if isempty(expr) || ~contains(expr,'('); continue; end
            try
                mName = strtrim(extractBefore(expr, '('));
                r = str2double(extractBetween(expr, '(', ',')); c = str2double(extractBetween(expr, ',', ')'));
                val = str2double(extractAfter(expr, '='));
                if isfield(pStrTmp, mName); pStrTmp.(mName)(r,c) = val; end
            catch
            end
        end
        c = cumsum([0, sz]); pLive(c(1)+1:c(2)) = log10(pStrTmp.r) + 1.0; pLive(c(6)+1:c(7)) = log10(pStrTmp.Z_transconjugation) + 2.0;
    end

    if ~(isHasFit && isJointMode)
        idx = 0;
        for i = 1:length(w_vec)
            prunedSegment = pLive(idx+1 : idx+sz(i));
            if sz(i) > 1
                flat = sort(abs(prunedSegment), 'descend'); w_c = max(1, w_vec(i)); thresh = flat(min(w_c, length(flat))); if w_vec(i) == 0; thresh = Inf; end
                prunedSegment(abs(prunedSegment) < thresh) = 0; pLive(idx+1 : idx+sz(i)) = prunedSegment; 
            elseif sz(i) == 1 && w_vec(i) == 0; pLive(idx+1 : idx+sz(i)) = -Inf; end
            idx = idx + sz(i);
        end
    end
    
    [Mats, pStr] = unpackAllMatrices(pLive, S, M, T, sz, icP, appData.datasets, S_obs, M_obs, T_obs);
    pStr = applyZeroLocationConstraints(pStr, ds); Mats = applyZeroLocationConstraints(Mats, ds);
    plotExplicitFit(appData.ax1, ds, pStr, S_obs, M_obs, T_obs, icP, 1.0, appData);
    plotExplicitFit(appData.ax2, ds, pStr, S_obs, M_obs, T_obs, icP, appData.ic_scale, appData);
    renderSingleHeatmap(fig, Mats, ds.names, S, M, T, maxS_hid, maxM_hid, maxT_hid, appData.dropMatrix.Value);
end



function plotExplicitFit(ax, ds, p, S_obs, M_obs, T_obs, icP, scale, appData)
    cla(ax); 
    t_g = linspace(ds.t(1), ds.t(end), 100); 
    y0 = zeros(p.S+p.M+p.T, 1); 
    for i = 1:S_obs
        if i <= length(icP.obs_y0); y0(i) = log(max(1e-6, icP.obs_y0(i)/1e8)); else; y0(i) = log(1e-6); end
    end
    if p.S > S_obs && isfield(p,'hidden_ICs') && ~isempty(p.hidden_ICs)
        y0(S_obs+1:p.S) = log(max(1e-6, p.hidden_ICs{1}.strains(:))); 
    end
    for i = 1:M_obs
        src = S_obs + i; 
        if src <= length(icP.obs_y0); y0(p.S+i) = log(max(1e-6, icP.obs_y0(src))); else; y0(p.S+i) = log(1e-2); end
    end
    if p.M > M_obs && isfield(p,'hidden_ICs') && ~isempty(p.hidden_ICs)
        y0(p.S+M_obs+1 : p.S+p.M) = log(max(1e-6, p.hidden_ICs{1}.metabolites(:))); 
    end
    for i = 1:T_obs
        src = S_obs + M_obs + i; 
        if src <= length(icP.obs_y0); y0(p.S+p.M+i) = log(max(1e-6, icP.obs_y0(src))); else; y0(p.S+p.M+i) = log(1e-3); end
    end
    if p.T > T_obs && isfield(p,'hidden_ICs') && ~isempty(p.hidden_ICs)
        y0(p.S+p.M+T_obs+1 : end) = log(max(1e-6, p.hidden_ICs{1}.toxins(:))); 
    end
    
    [~, ySim] = ode23s(@(t,y) explicitSystemODE(t,y,p), t_g, y0 * scale); 
    set(ax, 'YScale', 'log'); 
    clrs = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'};
    
    % --- FIXED DATA CLEANPASS PASS ---
    % Identifies and strips out the non-numeric 'Strains' text data column from the scatter plots
    isTextColMask = cellfun(@(x) ischar(x) || isstring(x), table2cell(table(ds.Y)));
    if any(isTextColMask)
        cleanDataMatrix = ds.Y;
        cleanDataMatrix(:, isTextColMask) = []; % Eliminates text entries before evaluation
    else
        cleanDataMatrix = ds.Y;
    end
    
    if scale == 1.0
        for v = 1:size(cleanDataMatrix, 2)
            plot(ax, ds.t, cleanDataMatrix(:, v), 'o', 'Color', clrs{mod(v-1,length(clrs))+1}, 'MarkerSize', 5); 
            hold(ax, 'on'); 
        end
        tStr = 'Least-Squares Prediction Fit'; 
    else
        hold(ax, 'on'); 
        tStr = sprintf('Alternative Mod Trajectory (Scaled: %.2fx)', scale); 
    end
    
    if size(ySim,1)==100 && all(isfinite(ySim(:)))
        yObs = [exp(ySim(:, 1:S_obs))*1e8, exp(ySim(:, p.S+1:p.S+M_obs)), exp(ySim(:, p.S+p.M+1:p.S+p.M+T_obs))]; 
        
        % Scrub labels text layout arrays similarly to synchronize legends
        legNames = ds.names(~strcmpi(ds.names, 'Strains'));
        for v = 1:size(yObs, 2)
            plot(ax, t_g, max(1e-5, yObs(:, v)), '-', 'Color', clrs{mod(v-1,length(clrs))+1}, 'LineWidth', 2); 
        end
        if appData.chkShowHiddenStr.Value && p.S > S_obs
            yH = exp(ySim(:, S_obs+1:p.S))*1e8;
            for k = 1:size(yH,2)
                plot(ax, t_g, max(1e-5, yH(:,k)), '--', 'Color', '#7E2F8E'); 
                legNames = [legNames, {sprintf('Hidden Strain %d',k)}]; 
            end
        end
        if appData.chkShowHiddenSub.Value && p.M > M_obs
            yH = exp(ySim(:, p.S+M_obs+1 : p.S+p.M));
            for k = 1:size(yH,2)
                plot(ax, t_g, max(1e-5, yH(:,k)), ':', 'Color', '#77AC30'); 
                legNames = [legNames, {sprintf('Hidden Metabolite %d',k)}]; 
            end
        end
        if appData.chkShowHiddenTox.Value && p.T > T_obs
            yH = exp(ySim(:, p.S+p.M+T_obs+1 : end));
            for k = 1:size(yH,2)
                plot(ax, t_g, max(1e-5, yH(:,k)), '-.', 'Color', '#A2142F'); 
                legNames = [legNames, {sprintf('Hidden Toxin %d',k)}]; 
            end
        end
    end
    title(ax, tStr); legend(ax, legNames, 'Location', 'best'); 
    ylim(ax, [1e-4, 5e9]); xlim(ax, [0, max(ds.t)]); hold(ax, 'off');
end

function renderSingleHeatmap(fig, Mats, names, S, M, T, Sh, Mh, Th, dropVal)
    appData = fig.UserData; gridPanel = appData.gridPanel; delete(gridPanel.Children);
    
    validNames = names(~strcmpi(names, 'Strains'));
    cellStrainsFilter = validNames(contains(validNames,'Recipient') | contains(validNames,'Donor') | contains(validNames,'Transconjugant') | contains(validNames,'Strain'));
    if isempty(cellStrainsFilter); cellStrainsFilter = {'Recipients', 'Donors', 'Transconjugants'}; end
    strains = cell(S, 1);
    for i = 1:S
        if i <= length(cellStrainsFilter); strains{i} = cellStrainsFilter{i};
        else; strains{i} = sprintf('HiddenStrain_%d', i-length(cellStrainsFilter)); end
    end
    
    nutrientSolutesFilter = validNames(contains(validNames,'Nutrient') | contains(validNames,'Substrate') | contains(validNames,'Metabolite'));
    if isempty(nutrientSolutesFilter); nutrientSolutesFilter = {'Nutrient_1'}; end
    subs = cell(M, 1);
    for j = 1:M
        if j <= length(nutrientSolutesFilter); subs{j} = nutrientSolutesFilter{j};
        else; subs{j} = sprintf('HiddenMetabolite_%d', j-length(nutrientSolutesFilter)); end
    end
    
    toxinsFilter = validNames(contains(validNames,'Toxin') | contains(validNames,'Inhibitor'));
    if isempty(toxinsFilter); toxinsFilter = {'Toxin_1'}; end
    toxins = cell(T, 1);
    for k = 1:T
        if k <= length(toxinsFilter); toxins{k} = toxinsFilter{k};
        else; toxins{k} = sprintf('HiddenToxin_%d', k-length(toxinsFilter)); end
    end
    
    matrixKeysList = {'r','k','P','K','c_consumption','Z_transconjugation','s_secretion','delta','m_supply','D_dilution','m_toxin_supply','d_toxin_decay'};
    activeMatrixKey = matrixKeysList{dropVal}; mData = Mats.(activeMatrixKey); 
    if iscolumn(mData); mData = mData'; end
    
    switch dropVal
        case {1, 2, 5}; y_l = strains; x_l = subs;
        case {3, 4, 7}; y_l = strains; x_l = toxins;
        case 6; y_l = strains; x_l = strains; 
        case 8; y_l = strains; x_l = {'Biomass Loss'};
        case {9, 10}; y_l = {'Inflow'}; x_l = subs;
        case {11, 12}; y_l = {'Inflow'}; x_l = toxins;
        otherwise; y_l = cellstr(string(1:size(mData,1))); x_l = cellstr(string(1:size(mData,2)));
    end
    
    tGrid = uitable(gridPanel, 'Position', [10 10 820 620]);
    tGrid.Data = mData; tGrid.RowName = y_l; tGrid.ColumnName = x_l;
    
    sStyle = uistyle; sStyle.BackgroundColor = [0.96 0.98 1.0]; tGrid.addStyle(sStyle);
    tGrid.CellSelectionCallback = @(src, ev) handleHeatmapClick(fig, src, ev, activeMatrixKey);
end

function handleHeatmapClick(fig, ~, ev, matrixName)
    if isempty(ev.Indices); return; end
    appData = fig.UserData; rowIdx = ev.Indices(1,1); colIdx = ev.Indices(1,2);
    tDSIdx = appData.dropDatasetData.Value; ds = appData.datasets{tDSIdx};
    
    promptStr = sprintf('Enter value to freeze entry %s(%d,%d) at constant during optimization:', matrixName, rowIdx, colIdx);
    valInput = inputdlg(promptStr, 'Matrix Cell Entry Lock', 1, {'0.00'});
    if isempty(valInput); return; end
    valNum = str2double(valInput{1}); if isnan(valNum); return; end
    
    if isfield(appData, 'local_fits') && length(appData.local_fits) >= tDSIdx && ~isempty(appData.local_fits{tDSIdx})
        lf = appData.local_fits{tDSIdx};
        [Mats, pStr] = unpackAllMatrices(lf.pRaw, lf.S, lf.M, lf.T, lf.sizes, lf.icProfile, {ds}, 3, 1, 1);
        if strcmp(matrixName, 'Z_transconjugation'); mKey = 'Z_transconjugation'; else; mKey = matrixName; end
        if isfield(pStr, mKey)
            pStr.(mKey)(rowIdx, colIdx) = valNum;
            c = cumsum([0, lf.sizes]);
            if strcmp(matrixName, 'r'); lf.pRaw(c(1)+1:c(2)) = log10(pStr.r) + 1.0;
            elseif strcmp(matrixName, 'Z_transconjugation'); lf.pRaw(c(6)+1:c(7)) = log10(pStr.Z_transconjugation) + 2.0; end
            appData.local_fits{tDSIdx} = lf;
        end
    end
    
    freezeLine = sprintf('# MatrixFreeze: %s(%d,%d)=%f', matrixName, rowIdx, colIdx, valNum);
    appData.txtPasteArea.Value = [appData.txtPasteArea.Value; {freezeLine}];
    fig.UserData = appData; parsePastedTextData(fig);
end
